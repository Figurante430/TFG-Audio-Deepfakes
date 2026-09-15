from __future__ import annotations

import argparse
import csv
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
        out = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        h = out.last_hidden_state
        feat_mask = self.wavlm._get_feature_vector_attention_mask(
            h.shape[1], attention_mask
        ).to(h.device)
        feat_mask_f = feat_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * feat_mask_f).sum(dim=1)
        pooled = pooled / feat_mask_f.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def get_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise RuntimeError("Formato de checkpoint no reconocido.")
    for key in ("model_state_dict", "state_dict", "model_state", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, nn.Module):
            return value.state_dict()
    if ckpt and any(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise RuntimeError("No encuentro el state_dict dentro del checkpoint.")


def normalize_state_dict_keys(state):
    out = {}
    for key, value in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("backbone."):
            k = "wavlm." + k[len("backbone."):]
        if k.startswith("classifier."):
            k = "head." + k[len("classifier."):]
        out[k] = value
    return out


def load_model(checkpoint_path: Path, device: torch.device):
    print(f"Cargando checkpoint:\n  {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = WavLMClassifier()
        state = normalize_state_dict_keys(get_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        important_missing = [k for k in missing if k.startswith("head.")]
        important_unexpected = [k for k in unexpected if k.startswith("head.")]
        if important_missing or important_unexpected:
            print("\nLa arquitectura de la cabeza no coincide.")
            print("Missing importantes:", important_missing)
            print("Unexpected importantes:", important_unexpected)
            print("\nPrimeras claves del checkpoint:")
            for k in list(state.keys())[:30]:
                print(" ", k)
            raise RuntimeError(
                "Checkpoint incompatible con la arquitectura esperada."
            )
        if missing:
            print(f"Aviso: {len(missing)} claves no encontradas en el checkpoint.")
        if unexpected:
            print(f"Aviso: {len(unexpected)} claves adicionales en el checkpoint.")
    model.to(device)
    model.eval()
    return model


def load_audio(path: Path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        raise RuntimeError(f"{path} está a {sr} Hz; esperaba {TARGET_SR} Hz.")
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) > MAX_SAMPLES:
        start = (len(wav) - MAX_SAMPLES) // 2
        wav = wav[start:start + MAX_SAMPLES]
        valid_len = MAX_SAMPLES
    else:
        valid_len = len(wav)
        if valid_len < MAX_SAMPLES:
            wav = np.pad(wav, (0, MAX_SAMPLES - valid_len), mode="constant")
    attention = np.zeros(MAX_SAMPLES, dtype=np.int64)
    attention[:valid_len] = 1
    return wav, attention


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


@torch.inference_mode()
def predict(model, wav_files, device, batch_size):
    rows = []
    total = len(wav_files)
    done = 0
    for paths in batched(wav_files, batch_size):
        xs, masks = [], []
        for path in paths:
            wav, mask = load_audio(path)
            xs.append(wav)
            masks.append(mask)
        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
        attention_mask = torch.tensor(np.stack(masks), dtype=torch.long, device=device)
        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, attention_mask)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        for path, prob in zip(paths, probs):
            rows.append({
                "path": str(path),
                "speaker_id": path.parent.name,
                "prob_fake": float(prob),
            })
        done += len(paths)
        print(f"\rProcesados: {done}/{total}", end="", flush=True)
    print()
    return rows


def print_report(rows, threshold):
    n = len(rows)
    tn = sum(row["prob_fake"] < threshold for row in rows)
    fp = n - tn
    specificity = tn / n if n else float("nan")
    fpr = fp / n if n else float("nan")
    probs = np.array([row["prob_fake"] for row in rows], dtype=float)

    print("\n" + "=" * 68)
    print("COMMON VOICE — EXTERNAL REAL TEST")
    print("=" * 68)
    print(f"Audios reales externos : {n}")
    print(f"Threshold fijo         : {threshold:.6f}")
    print(f"TN (real -> real)      : {tn}")
    print(f"FP (real -> fake)      : {fp}")
    print(f"Specificity / TNR      : {specificity:.4f}")
    print(f"False Positive Rate    : {fpr:.4f}")
    print(f"Prob fake media        : {probs.mean():.6f}")
    print(f"Prob fake mediana      : {np.median(probs):.6f}")
    print(f"Prob fake mínima       : {probs.min():.6f}")
    print(f"Prob fake máxima       : {probs.max():.6f}")

    print("\nPor hablante externo:")
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    for speaker in sorted(groups):
        sr = groups[speaker]
        s_tn = sum(r["prob_fake"] < threshold for r in sr)
        s_fp = len(sr) - s_tn
        s_spec = s_tn / len(sr)
        print(f"  {speaker:20s} N={len(sr):3d}  TN={s_tn:3d}  FP={s_fp:3d}  Spec={s_spec:.4f}")

    false_positives = sorted(
        (r for r in rows if r["prob_fake"] >= threshold),
        key=lambda r: r["prob_fake"],
        reverse=True,
    )
    if false_positives:
        print("\nFalsos positivos:")
        for row in false_positives[:20]:
            print(f"  p_fake={row['prob_fake']:.6f}  {row['path']}")
    else:
        print("\nNo hay falsos positivos con este threshold.")
    print("=" * 68)


def save_csv(rows, output, threshold):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "speaker_id", "prob_fake",
                "threshold", "pred_label", "pred_class"
            ],
        )
        writer.writeheader()
        for row in rows:
            pred = 1 if row["prob_fake"] >= threshold else 0
            writer.writerow({
                **row,
                "threshold": threshold,
                "pred_label": pred,
                "pred_class": "fake" if pred else "real",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold fijo obtenido previamente en validation. No calibrar con Common Voice.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path(r"C:\Users\gonza\TFG\Deepfake-Detection\external_test\real"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\gonza\TFG\Deepfake-Detection\external_test\results_external_real.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    wav_files = sorted(args.real_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No encuentro WAV en {args.real_dir}")

    print(f"Encontrados {len(wav_files)} WAV externos.")
    if len(wav_files) != 100:
        print(f"AVISO: esperábamos 100 audios, pero hay {len(wav_files)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = load_model(args.checkpoint, device)
    rows = predict(model, wav_files, device, args.batch_size)
    print_report(rows, args.threshold)
    save_csv(rows, args.output, args.threshold)
    print(f"\nResultados guardados en:\n  {args.output}")


if __name__ == "__main__":
    main()
