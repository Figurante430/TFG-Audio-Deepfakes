from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WavLMModel


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16000
SECONDS = 4
MAX_SAMPLES = TARGET_SR * SECONDS

SEEDS = [42, 123, 2026]
HOLDOUTS = [
    "confucius4",
    "cosyvoice",
    "knnvc",
    "openvoice",
    "qwen",
    "rvc",
    "voxcpm2",
]

# Contrafactuales
TARGET_RMS_DBFS = -25.0
GATE_DBFS = -45.0
LOWPASS_HZ = 4000.0
LOWPASS_TAPS = 513

VARIANTS = [
    "original",
    "rms_m25",
    "gate_m45",
    "rms_m25_gate_m45",
    "lowpass_4k",
]


# ============================================================
# MODELO
# ============================================================

class WavLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(MODEL_NAME)
        hidden = self.wavlm.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )

    def forward(self, input_values, attention_mask):
        out = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        h = out.last_hidden_state

        feat_mask = self.wavlm._get_feature_vector_attention_mask(
            h.shape[1],
            attention_mask,
        ).to(h.device)

        feat_mask_f = feat_mask.unsqueeze(-1).to(h.dtype)

        pooled = (h * feat_mask_f).sum(dim=1)
        denom = feat_mask_f.sum(dim=1).clamp_min(1.0)
        pooled = pooled / denom

        return self.head(pooled).squeeze(-1)


def get_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise RuntimeError("Formato de checkpoint no reconocido.")

    for key in (
        "model_state_dict",
        "state_dict",
        "model_state",
        "model",
    ):
        value = ckpt.get(key)

        if isinstance(value, dict):
            return value

        if isinstance(value, nn.Module):
            return value.state_dict()

    if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise RuntimeError(
        "No encuentro el state_dict dentro del checkpoint."
    )


def normalize_state_dict_keys(state):
    out = {}

    for key, value in state.items():
        k = key

        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]

        if k.startswith("backbone."):
            k = "wavlm." + k[len("backbone."):]

        if k.startswith("classifier."):
            k = "head." + k[len("classifier."):]

        out[k] = value

    return out


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = WavLMClassifier()

        state = normalize_state_dict_keys(
            get_state_dict(checkpoint)
        )

        missing, unexpected = model.load_state_dict(
            state,
            strict=False,
        )

        important_missing = [
            k for k in missing if k.startswith("head.")
        ]

        important_unexpected = [
            k for k in unexpected
            if k.startswith("head.") or k.startswith("classifier.")
        ]

        if important_missing or important_unexpected:
            raise RuntimeError(
                f"Checkpoint incompatible: {checkpoint_path}\n"
                f"Missing: {important_missing}\n"
                f"Unexpected: {important_unexpected}"
            )

    model.to(device)
    model.eval()

    return model


# ============================================================
# AUDIO / TRANSFORMACIONES
# ============================================================

def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def rms(x):
    if len(x) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.asarray(x, dtype=np.float64) ** 2
            )
        )
    )


def normalize_rms(x, target_dbfs=TARGET_RMS_DBFS):
    x = np.asarray(x, dtype=np.float32).copy()

    current = rms(x)

    if current <= 1e-12:
        return x

    target = db_to_amp(target_dbfs)
    gain = target / current

    y = x * gain

    # Evitamos clipping artificial.
    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.98:
        y = y * (0.98 / peak)

    return y.astype(np.float32)


def noise_gate(x, threshold_dbfs=GATE_DBFS):
    """
    Contrafactual del noise floor / near-zero ratio.

    No recorta ni desplaza el audio. Solo fuerza a cero
    muestras de muy baja amplitud. Esto evita cambiar el
    alineamiento temporal del clip.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    threshold = db_to_amp(threshold_dbfs)

    x[np.abs(x) < threshold] = 0.0

    return x


def fir_lowpass(
    x,
    cutoff_hz=LOWPASS_HZ,
    sr=TARGET_SR,
    taps=LOWPASS_TAPS,
):
    """
    FIR windowed-sinc, sin scipy/librosa.
    """
    x = np.asarray(x, dtype=np.float32)

    if taps % 2 == 0:
        taps += 1

    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / sr

    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.hamming(taps)
    h /= np.sum(h)

    y = np.convolve(
        x.astype(np.float64),
        h.astype(np.float64),
        mode="same",
    )

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.999:
        y = y * (0.999 / peak)

    return y.astype(np.float32)


def transform_audio(x, variant):
    if variant == "original":
        return np.asarray(x, dtype=np.float32)

    if variant == "rms_m25":
        return normalize_rms(x)

    if variant == "gate_m45":
        return noise_gate(x)

    if variant == "rms_m25_gate_m45":
        y = normalize_rms(x)
        y = noise_gate(y)
        return y

    if variant == "lowpass_4k":
        return fir_lowpass(x)

    raise ValueError(
        f"Variante desconocida: {variant}"
    )


def load_source_audio(path: Path):
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = x.mean(axis=1)

    if sr != TARGET_SR:
        raise RuntimeError(
            f"{path} está a {sr} Hz; esperaba {TARGET_SR} Hz."
        )

    return np.asarray(x, dtype=np.float32)


def model_window(x):
    """
    Exactamente el mismo crop/pad de evaluación:
    - >4 s: crop central
    - <4 s: zero-pad al final
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) > MAX_SAMPLES:
        start = (len(x) - MAX_SAMPLES) // 2
        y = x[start:start + MAX_SAMPLES]
        valid_len = MAX_SAMPLES
    else:
        valid_len = len(x)

        if valid_len < MAX_SAMPLES:
            y = np.pad(
                x,
                (0, MAX_SAMPLES - valid_len),
                mode="constant",
            )
        else:
            y = x

    mask = np.zeros(
        MAX_SAMPLES,
        dtype=np.int64,
    )
    mask[:valid_len] = 1

    return y.astype(np.float32), mask


def prepare_variants(
    wav_files,
    output_audio_root: Path,
):
    """
    Genera las copias diagnósticas y deja en RAM
    exactamente las ventanas que verá el modelo.
    """
    prepared = {
        variant: []
        for variant in VARIANTS
    }

    for i, path in enumerate(wav_files, 1):
        source = load_source_audio(path)

        for variant in VARIANTS:
            transformed = transform_audio(
                source,
                variant,
            )

            out_path = (
                output_audio_root
                / variant
                / path.parent.name
                / path.name
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # PCM16, igual que el dataset normalizado.
            sf.write(
                out_path,
                transformed,
                TARGET_SR,
                subtype="PCM_16",
            )

            # Releer desde PCM16 para evaluar exactamente
            # el fichero que hemos guardado.
            saved, saved_sr = sf.read(
                out_path,
                dtype="float32",
                always_2d=False,
            )

            if saved.ndim == 2:
                saved = saved.mean(axis=1)

            if saved_sr != TARGET_SR:
                raise RuntimeError(
                    f"SR inesperado en {out_path}"
                )

            window, mask = model_window(
                np.asarray(saved, dtype=np.float32)
            )

            prepared[variant].append(
                {
                    "path": out_path,
                    "filename": path.name,
                    "speaker_id": path.parent.name,
                    "wav": window,
                    "mask": mask,
                }
            )

        print(
            f"\rPreparando contrafactuales: {i}/{len(wav_files)}",
            end="",
            flush=True,
        )

    print()

    return prepared


# ============================================================
# EVALUACIÓN
# ============================================================

def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


@torch.inference_mode()
def predict(
    model,
    items,
    device,
    batch_size,
):
    rows = []

    for batch in batched(
        items,
        batch_size,
    ):
        x = torch.tensor(
            np.stack(
                [item["wav"] for item in batch]
            ),
            dtype=torch.float32,
            device=device,
        )

        attention_mask = torch.tensor(
            np.stack(
                [item["mask"] for item in batch]
            ),
            dtype=torch.long,
            device=device,
        )

        use_amp = device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                x,
                attention_mask,
            )

        probs = torch.sigmoid(
            logits.float()
        ).cpu().numpy()

        for item, prob in zip(
            batch,
            probs,
        ):
            rows.append(
                {
                    "path": str(item["path"]),
                    "filename": item["filename"],
                    "speaker_id": item["speaker_id"],
                    "prob_fake": float(prob),
                }
            )

    return rows


def load_threshold(metrics_path: Path):
    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if "calibrated_threshold" in data:
        return float(
            data["calibrated_threshold"]
        )

    summary = data.get(
        "summary_row",
        {}
    )

    if "threshold_val" in summary:
        return float(
            summary["threshold_val"]
        )

    raise RuntimeError(
        f"No encuentro calibrated_threshold en {metrics_path}"
    )


def calc_metrics(
    predictions,
    threshold,
):
    n = len(predictions)

    tn = sum(
        row["prob_fake"] < threshold
        for row in predictions
    )

    fp = n - tn

    return {
        "n": n,
        "tn": tn,
        "fp": fp,
        "specificity": tn / n,
        "fpr": fp / n,
    }


def mean_std(values):
    values = np.asarray(
        list(values),
        dtype=float,
    )

    return (
        float(np.mean(values)),
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0,
    )


def write_csv(
    path: Path,
    rows,
    fieldnames=None,
):
    rows = list(rows)

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"C:\Users\gonza\TFG\Deepfake-Detection"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    root = args.root

    source_real_dir = (
        root
        / "external_test"
        / "real"
    )

    out_root = (
        root
        / "external_test"
        / "counterfactual_shortcuts"
    )

    out_audio_root = (
        out_root
        / "audio"
    )

    wav_files = sorted(
        source_real_dir.rglob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No hay WAV en {source_real_dir}"
        )

    print(
        f"Audios externos encontrados: {len(wav_files)}"
    )

    print()
    print(
        "Generando variantes diagnósticas..."
    )

    prepared = prepare_variants(
        wav_files,
        out_audio_root,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")

    run_rows = []
    speaker_rows = []
    prediction_rows = []

    total_models = len(HOLDOUTS) * len(SEEDS)
    model_index = 0

    for holdout in HOLDOUTS:
        for seed in SEEDS:
            model_index += 1

            exp_dir = (
                root
                / "experiments"
                / "logo_wavlm_3seeds"
                / f"holdout_{holdout}"
                / f"seed_{seed}"
            )

            checkpoint = (
                exp_dir
                / "best_model.pt"
            )

            metrics_json = (
                exp_dir
                / "metrics.json"
            )

            threshold = load_threshold(
                metrics_json
            )

            print()
            print("=" * 80)
            print(
                f"MODELO {model_index}/{total_models} | "
                f"{holdout} | seed={seed} | "
                f"thr={threshold:.6f}"
            )
            print("=" * 80)

            model = load_model(
                checkpoint,
                device,
            )

            for variant in VARIANTS:
                preds = predict(
                    model,
                    prepared[variant],
                    device,
                    args.batch_size,
                )

                metrics = calc_metrics(
                    preds,
                    threshold,
                )

                print(
                    f"  {variant:20s} "
                    f"Spec={metrics['specificity']:.4f} "
                    f"FP={metrics['fp']:3d}"
                )

                run_rows.append(
                    {
                        "held_out_generator": holdout,
                        "seed": seed,
                        "threshold": threshold,
                        "variant": variant,
                        **metrics,
                    }
                )

                by_speaker = defaultdict(list)

                for row in preds:
                    by_speaker[
                        row["speaker_id"]
                    ].append(row)

                    prediction_rows.append(
                        {
                            "held_out_generator": holdout,
                            "seed": seed,
                            "threshold": threshold,
                            "variant": variant,
                            **row,
                            "pred_fake": int(
                                row["prob_fake"] >= threshold
                            ),
                        }
                    )

                for speaker, rows in sorted(
                    by_speaker.items()
                ):
                    sm = calc_metrics(
                        rows,
                        threshold,
                    )

                    speaker_rows.append(
                        {
                            "held_out_generator": holdout,
                            "seed": seed,
                            "variant": variant,
                            "speaker_id": speaker,
                            **sm,
                        }
                    )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ========================================================
    # RESÚMENES
    # ========================================================

    write_csv(
        out_root
        / "summary_all_runs.csv",
        run_rows,
    )

    write_csv(
        out_root
        / "summary_by_speaker_all_runs.csv",
        speaker_rows,
    )

    write_csv(
        out_root
        / "predictions_all_runs.csv",
        prediction_rows,
    )

    # Promedio de los 21 modelos por variante
    variant_groups = defaultdict(list)

    for row in run_rows:
        variant_groups[
            row["variant"]
        ].append(row)

    variant_summary = []

    original_mean = None

    for variant in VARIANTS:
        rows = variant_groups[variant]

        specs = [
            float(r["specificity"])
            for r in rows
        ]

        fprs = [
            float(r["fpr"])
            for r in rows
        ]

        spec_mean, spec_std = mean_std(
            specs
        )

        fpr_mean, fpr_std = mean_std(
            fprs
        )

        if variant == "original":
            original_mean = spec_mean

        variant_summary.append(
            {
                "variant": variant,
                "n_models": len(rows),
                "specificity_mean": spec_mean,
                "specificity_std": spec_std,
                "fpr_mean": fpr_mean,
                "fpr_std": fpr_std,
            }
        )

    for row in variant_summary:
        row[
            "delta_specificity_vs_original"
        ] = (
            row["specificity_mean"]
            - original_mean
        )

    write_csv(
        out_root
        / "summary_variants_mean_std.csv",
        variant_summary,
    )

    # Promedio por speaker y variante
    sv_groups = defaultdict(list)

    for row in speaker_rows:
        sv_groups[
            (
                row["variant"],
                row["speaker_id"],
            )
        ].append(row)

    speaker_variant_summary = []

    speakers = sorted(
        set(
            row["speaker_id"]
            for row in speaker_rows
        )
    )

    original_speaker_means = {}

    for speaker in speakers:
        rows = sv_groups[
            ("original", speaker)
        ]

        original_speaker_means[
            speaker
        ] = float(
            np.mean(
                [
                    float(r["specificity"])
                    for r in rows
                ]
            )
        )

    for variant in VARIANTS:
        for speaker in speakers:
            rows = sv_groups[
                (variant, speaker)
            ]

            specs = [
                float(r["specificity"])
                for r in rows
            ]

            mean, std = mean_std(
                specs
            )

            speaker_variant_summary.append(
                {
                    "variant": variant,
                    "speaker_id": speaker,
                    "n_models": len(rows),
                    "specificity_mean": mean,
                    "specificity_std": std,
                    "delta_specificity_vs_original": (
                        mean
                        - original_speaker_means[speaker]
                    ),
                }
            )

    write_csv(
        out_root
        / "summary_speakers_variants.csv",
        speaker_variant_summary,
    )

    # Efecto individual por fichero:
    # fake-vote-rate entre los 21 modelos para cada variante.
    file_groups = defaultdict(list)

    for row in prediction_rows:
        file_groups[
            (
                row["variant"],
                row["speaker_id"],
                row["filename"],
            )
        ].append(row)

    file_summary = []

    for (
        variant,
        speaker,
        filename,
    ), rows in file_groups.items():

        vote_rate = float(
            np.mean(
                [
                    int(r["pred_fake"])
                    for r in rows
                ]
            )
        )

        prob_mean = float(
            np.mean(
                [
                    float(r["prob_fake"])
                    for r in rows
                ]
            )
        )

        file_summary.append(
            {
                "variant": variant,
                "speaker_id": speaker,
                "filename": filename,
                "n_models": len(rows),
                "fake_vote_rate": vote_rate,
                "prob_fake_mean": prob_mean,
            }
        )

    write_csv(
        out_root
        / "summary_by_file_variant.csv",
        file_summary,
    )

    # ========================================================
    # REPORTE
    # ========================================================

    report = (
        out_root
        / "report_counterfactual.txt"
    )

    with report.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "COUNTERFACTUAL SHORTCUT TEST — COMMON VOICE REAL\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Audios: {len(wav_files)}\n"
        )
        f.write(
            f"Modelos: {total_models}\n"
        )
        f.write(
            "Thresholds: calibrados exclusivamente con validation de cada modelo\n\n"
        )

        f.write(
            "ESPECIFICIDAD MEDIA DE LOS 21 MODELOS\n"
        )
        f.write("-" * 80 + "\n")

        for row in variant_summary:
            f.write(
                f"{row['variant']:22s} "
                f"Spec={row['specificity_mean']:.4f} "
                f"± {row['specificity_std']:.4f} "
                f"FPR={row['fpr_mean']:.4f} "
                f"delta={row['delta_specificity_vs_original']:+.4f}\n"
            )

        f.write("\n")
        f.write(
            "ESPECIFICIDAD POR SPEAKER Y VARIANTE\n"
        )
        f.write("-" * 80 + "\n")

        for variant in VARIANTS:
            f.write(
                f"\n[{variant}]\n"
            )

            rows = [
                r
                for r in speaker_variant_summary
                if r["variant"] == variant
            ]

            for row in rows:
                f.write(
                    f"{row['speaker_id']:18s} "
                    f"Spec={row['specificity_mean']:.4f} "
                    f"± {row['specificity_std']:.4f} "
                    f"delta={row['delta_specificity_vs_original']:+.4f}\n"
                )

        f.write("\n")
        f.write(
            "INTERPRETACIÓN DE LAS VARIANTES\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            "rms_m25: normaliza el RMS a -25 dBFS, con protección contra clipping.\n"
        )
        f.write(
            "gate_m45: fuerza a cero muestras por debajo de -45 dBFS sin cambiar duración.\n"
        )
        f.write(
            "rms_m25_gate_m45: aplica ambas operaciones.\n"
        )
        f.write(
            "lowpass_4k: FIR paso bajo a 4 kHz para reducir contenido de alta frecuencia.\n"
        )
        f.write(
            "original: copia PCM16 del audio externo sin transformación diagnóstica.\n"
        )
        f.write(
            "\nUn delta positivo indica que la transformación redujo falsos positivos.\n"
        )
        f.write(
            "Este experimento es diagnóstico: no sustituye al test externo original.\n"
        )

    print()
    print("=" * 80)
    print("RESULTADO GLOBAL")
    print("=" * 80)

    for row in variant_summary:
        print(
            f"{row['variant']:22s} "
            f"Spec={row['specificity_mean']:.4f} "
            f"± {row['specificity_std']:.4f} "
            f"delta={row['delta_specificity_vs_original']:+.4f}"
        )

    print()
    print(
        "Reporte guardado en:"
    )
    print(
        f"  {report}"
    )


if __name__ == "__main__":
    main()
