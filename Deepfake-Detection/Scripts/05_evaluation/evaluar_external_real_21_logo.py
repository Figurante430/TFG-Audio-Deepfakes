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
            k for k in missing
            if k.startswith("head.")
        ]

        important_unexpected = [
            k for k in unexpected
            if (
                k.startswith("head.")
                or k.startswith("classifier.")
            )
        ]

        if important_missing or important_unexpected:
            print("\nCheckpoint incompatible:")
            print("Missing importantes:", important_missing)
            print("Unexpected importantes:", important_unexpected)
            raise RuntimeError(
                f"Checkpoint incompatible: {checkpoint_path}"
            )

    model.to(device)
    model.eval()

    return model


def load_audio(path: Path):
    wav, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        raise RuntimeError(
            f"{path} está a {sr} Hz; esperaba {TARGET_SR} Hz."
        )

    wav = np.asarray(wav, dtype=np.float32)

    if len(wav) > MAX_SAMPLES:
        start = (len(wav) - MAX_SAMPLES) // 2
        wav = wav[start:start + MAX_SAMPLES]
        valid_len = MAX_SAMPLES
    else:
        valid_len = len(wav)

        if valid_len < MAX_SAMPLES:
            wav = np.pad(
                wav,
                (0, MAX_SAMPLES - valid_len),
                mode="constant",
            )

    attention = np.zeros(
        MAX_SAMPLES,
        dtype=np.int64,
    )
    attention[:valid_len] = 1

    return wav, attention


def preload_external_real(wav_files):
    """
    Carga los 100 audios una sola vez en RAM para no releerlos
    del disco en los 21 modelos.
    """
    loaded = []

    for i, path in enumerate(wav_files, 1):
        wav, mask = load_audio(path)

        loaded.append(
            {
                "path": path,
                "speaker_id": path.parent.name,
                "wav": wav,
                "mask": mask,
            }
        )

        print(
            f"\rPrecargando audios: {i}/{len(wav_files)}",
            end="",
            flush=True,
        )

    print()
    return loaded


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


@torch.inference_mode()
def predict(
    model,
    loaded_items,
    device,
    batch_size,
):
    predictions = []

    done = 0
    total = len(loaded_items)

    for batch in batched(
        loaded_items,
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
            predictions.append(
                {
                    "path": str(item["path"]),
                    "filename": item["path"].name,
                    "speaker_id": item["speaker_id"],
                    "prob_fake": float(prob),
                }
            )

        done += len(batch)

        print(
            f"\rInferencia: {done}/{total}",
            end="",
            flush=True,
        )

    print()
    return predictions


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


def compute_run_metrics(
    predictions,
    threshold,
):
    n = len(predictions)

    tn = sum(
        p["prob_fake"] < threshold
        for p in predictions
    )
    fp = n - tn

    probs = np.array(
        [p["prob_fake"] for p in predictions],
        dtype=np.float64,
    )

    return {
        "n": n,
        "tn": tn,
        "fp": fp,
        "specificity": tn / n,
        "fpr": fp / n,
        "prob_fake_mean": float(probs.mean()),
        "prob_fake_median": float(np.median(probs)),
        "prob_fake_min": float(probs.min()),
        "prob_fake_max": float(probs.max()),
    }


def sample_std(values):
    values = list(values)

    if len(values) <= 1:
        return 0.0

    return float(
        np.std(
            np.array(values, dtype=float),
            ddof=1,
        )
    )


def write_csv(
    path: Path,
    rows,
    fieldnames=None,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(rows)

    if not rows:
        return

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


def run_one(
    root: Path,
    output_root: Path,
    holdout: str,
    seed: int,
    loaded_items,
    device,
    batch_size,
    force=False,
):
    exp_dir = (
        root
        / "experiments"
        / "logo_wavlm_3seeds"
        / f"holdout_{holdout}"
        / f"seed_{seed}"
    )

    checkpoint = (
        exp_dir / "best_model.pt"
    )

    metrics_path = (
        exp_dir / "metrics.json"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No existe: {checkpoint}"
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No existe: {metrics_path}"
        )

    run_dir = (
        output_root
        / f"holdout_{holdout}"
        / f"seed_{seed}"
    )

    run_json = (
        run_dir / "external_real_metrics.json"
    )

    pred_csv = (
        run_dir / "predictions.csv"
    )

    if (
        not force
        and run_json.exists()
        and pred_csv.exists()
    ):
        print(
            f"[SKIP] {holdout} seed={seed} "
            "(ya existe)"
        )

        with run_json.open(
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        with pred_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            predictions = list(
                csv.DictReader(f)
            )

        for row in predictions:
            row["prob_fake"] = float(
                row["prob_fake"]
            )

        return metrics, predictions

    threshold = load_threshold(
        metrics_path
    )

    print()
    print("=" * 72)
    print(
        f"HOLDOUT={holdout} | SEED={seed} "
        f"| threshold={threshold:.6f}"
    )
    print("=" * 72)

    model = load_model(
        checkpoint,
        device,
    )

    predictions = predict(
        model,
        loaded_items,
        device,
        batch_size,
    )

    metrics = compute_run_metrics(
        predictions,
        threshold,
    )

    metrics.update(
        {
            "held_out_generator": holdout,
            "seed": seed,
            "threshold": threshold,
            "checkpoint": str(checkpoint),
        }
    )

    for row in predictions:
        row["held_out_generator"] = holdout
        row["seed"] = seed
        row["threshold"] = threshold
        row["pred_label"] = (
            1
            if row["prob_fake"] >= threshold
            else 0
        )
        row["pred_class"] = (
            "fake"
            if row["pred_label"] == 1
            else "real"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        pred_csv,
        predictions,
    )

    with run_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"TN={metrics['tn']}  "
        f"FP={metrics['fp']}  "
        f"Specificity={metrics['specificity']:.4f}  "
        f"FPR={metrics['fpr']:.4f}"
    )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics, predictions


def build_speaker_rows(
    holdout,
    seed,
    threshold,
    predictions,
):
    groups = defaultdict(list)

    for row in predictions:
        groups[
            row["speaker_id"]
        ].append(row)

    rows = []

    for speaker in sorted(groups):
        items = groups[speaker]
        n = len(items)

        tn = sum(
            item["prob_fake"] < threshold
            for item in items
        )
        fp = n - tn

        rows.append(
            {
                "held_out_generator": holdout,
                "seed": seed,
                "speaker_id": speaker,
                "n": n,
                "tn": tn,
                "fp": fp,
                "specificity": tn / n,
                "fpr": fp / n,
            }
        )

    return rows


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

    parser.add_argument(
        "--force",
        action="store_true",
        help="Repite también los runs ya completados.",
    )

    args = parser.parse_args()

    root = args.root

    real_dir = (
        root
        / "external_test"
        / "real"
    )

    output_root = (
        root
        / "external_test"
        / "logo_wavlm_21_external_real"
    )

    wav_files = sorted(
        real_dir.rglob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No hay WAV en {real_dir}"
        )

    print(
        f"Audios externos encontrados: {len(wav_files)}"
    )

    if len(wav_files) != 100:
        print(
            "AVISO: se esperaban 100 WAV."
        )

    loaded_items = preload_external_real(
        wav_files
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")

    run_rows = []
    speaker_rows = []
    all_prediction_rows = []

    for holdout in HOLDOUTS:
        for seed in SEEDS:
            metrics, predictions = run_one(
                root=root,
                output_root=output_root,
                holdout=holdout,
                seed=seed,
                loaded_items=loaded_items,
                device=device,
                batch_size=args.batch_size,
                force=args.force,
            )

            run_rows.append(
                {
                    "held_out_generator": holdout,
                    "seed": seed,
                    "threshold": metrics["threshold"],
                    "n": metrics["n"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "specificity": metrics["specificity"],
                    "fpr": metrics["fpr"],
                    "prob_fake_mean": metrics["prob_fake_mean"],
                    "prob_fake_median": metrics["prob_fake_median"],
                    "prob_fake_min": metrics["prob_fake_min"],
                    "prob_fake_max": metrics["prob_fake_max"],
                }
            )

            speaker_rows.extend(
                build_speaker_rows(
                    holdout,
                    seed,
                    metrics["threshold"],
                    predictions,
                )
            )

            for row in predictions:
                all_prediction_rows.append(
                    dict(row)
                )

    write_csv(
        output_root / "summary_all_runs.csv",
        run_rows,
    )

    write_csv(
        output_root / "summary_by_speaker_all_runs.csv",
        speaker_rows,
    )

    write_csv(
        output_root / "predictions_all_runs.csv",
        all_prediction_rows,
    )

    # Resumen mean ± std por generador holdout
    grouped = defaultdict(list)

    for row in run_rows:
        grouped[
            row["held_out_generator"]
        ].append(row)

    mean_std_rows = []

    for holdout in HOLDOUTS:
        rows = grouped[holdout]

        specs = [
            float(r["specificity"])
            for r in rows
        ]

        fprs = [
            float(r["fpr"])
            for r in rows
        ]

        thresholds = [
            float(r["threshold"])
            for r in rows
        ]

        mean_std_rows.append(
            {
                "held_out_generator": holdout,
                "n_seeds": len(rows),
                "specificity_mean": float(np.mean(specs)),
                "specificity_std": sample_std(specs),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std": sample_std(fprs),
                "threshold_mean": float(np.mean(thresholds)),
                "threshold_std": sample_std(thresholds),
            }
        )

    write_csv(
        output_root / "summary_mean_std.csv",
        mean_std_rows,
    )

    # Resumen por speaker promediando los 21 modelos
    speaker_grouped = defaultdict(list)

    for row in speaker_rows:
        speaker_grouped[
            row["speaker_id"]
        ].append(row)

    speaker_mean_rows = []

    for speaker in sorted(speaker_grouped):
        rows = speaker_grouped[speaker]

        specs = [
            float(r["specificity"])
            for r in rows
        ]

        fprs = [
            float(r["fpr"])
            for r in rows
        ]

        speaker_mean_rows.append(
            {
                "speaker_id": speaker,
                "n_models": len(rows),
                "specificity_mean": float(np.mean(specs)),
                "specificity_std": sample_std(specs),
                "fpr_mean": float(np.mean(fprs)),
                "fpr_std": sample_std(fprs),
            }
        )

    write_csv(
        output_root / "summary_speakers_mean_std.csv",
        speaker_mean_rows,
    )

    # Consenso por audio: cuántos de los 21 modelos lo marcan fake
    consensus = defaultdict(list)

    for row in all_prediction_rows:
        consensus[
            row["path"]
        ].append(row)

    consensus_rows = []

    for path, rows in sorted(consensus.items()):
        probs = np.array(
            [
                float(r["prob_fake"])
                for r in rows
            ],
            dtype=float,
        )

        n_fake = sum(
            int(r["pred_label"]) == 1
            for r in rows
        )

        consensus_rows.append(
            {
                "path": path,
                "filename": rows[0]["filename"],
                "speaker_id": rows[0]["speaker_id"],
                "n_models": len(rows),
                "n_pred_fake": n_fake,
                "fake_vote_rate": n_fake / len(rows),
                "prob_fake_mean": float(probs.mean()),
                "prob_fake_median": float(np.median(probs)),
                "prob_fake_min": float(probs.min()),
                "prob_fake_max": float(probs.max()),
            }
        )

    consensus_rows.sort(
        key=lambda r: (
            r["fake_vote_rate"],
            r["prob_fake_mean"],
        ),
        reverse=True,
    )

    write_csv(
        output_root / "consensus_by_file.csv",
        consensus_rows,
    )

    print()
    print("=" * 72)
    print("RESUMEN FINAL")
    print("=" * 72)

    for row in mean_std_rows:
        print(
            f"{row['held_out_generator']:12s} "
            f"Spec={row['specificity_mean']:.4f}"
            f" ± {row['specificity_std']:.4f}  "
            f"FPR={row['fpr_mean']:.4f}"
            f" ± {row['fpr_std']:.4f}"
        )

    print()
    print("Promedio por hablante externo:")

    for row in speaker_mean_rows:
        print(
            f"{row['speaker_id']:18s} "
            f"Spec={row['specificity_mean']:.4f}"
            f" ± {row['specificity_std']:.4f}"
        )

    print()
    print(
        "Resultados guardados en:"
    )
    print(
        f"  {output_root}"
    )


if __name__ == "__main__":
    main()
