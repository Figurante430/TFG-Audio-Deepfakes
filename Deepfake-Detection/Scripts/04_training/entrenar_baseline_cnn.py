from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
METADATA_CSV = ROOT / "data_normalized" / "metadata_normalized.csv"

OUTPUT_DIR = ROOT / "experiments" / "baseline_logmel_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = OUTPUT_DIR / "best_model.pt"
HISTORY_CSV = OUTPUT_DIR / "history.csv"
TEST_METRICS_JSON = OUTPUT_DIR / "test_metrics.json"
TEST_PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"

SEED = 42

SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows: empezar por 0 para evitar problemas de multiprocessing

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METADATA
# ============================================================

def read_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro:\n{METADATA_CSV}"
        )

    with METADATA_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("metadata_normalized.csv está vacío.")

    required = {
        "path",
        "label",
        "split",
        "generator",
        "category",
        "speaker_id",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "Faltan columnas: " + ", ".join(sorted(missing))
        )

    # Solo filas con split válido
    rows = [
        row for row in rows
        if row["split"] in {"train", "val", "test"}
    ]

    return rows


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return ROOT / path


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, path: Path):
        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )

        # soundfile -> [samples, channels]
        # Convertimos a mono
        audio = audio.mean(axis=1)

        waveform = torch.from_numpy(audio).float()

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sr,
                SAMPLE_RATE,
            ).squeeze(0)

        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor):
        n = waveform.numel()

        if n > NUM_SAMPLES:
            max_start = n - NUM_SAMPLES

            if self.training:
                start = int(
                    torch.randint(
                        low=0,
                        high=max_start + 1,
                        size=(1,),
                    ).item()
                )
            else:
                start = max_start // 2

            waveform = waveform[
                start:start + NUM_SAMPLES
            ]

        elif n < NUM_SAMPLES:
            missing = NUM_SAMPLES - n

            # Padding simétrico
            left = missing // 2
            right = missing - left

            waveform = F.pad(
                waveform,
                (left, right),
            )

        return waveform

    def __getitem__(self, index):
        row = self.rows[index]
        path = resolve_path(row["path"])

        waveform = self._load_audio(path)
        waveform = self._crop_or_pad(waveform)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        return {
            "waveform": waveform,
            "label": label,
            "path": row["path"],
            "generator": row["generator"],
            "speaker_id": row["speaker_id"],
        }


# ============================================================
# LOG-MEL
# ============================================================

class LogMelFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

        self.db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

    def forward(self, waveform):
        # waveform: [B, T]
        x = self.mel(waveform)
        x = self.db(x)

        # Normalización por muestra
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        x = (x - mean) / std

        # [B, 1, n_mels, frames]
        return x.unsqueeze(1)


# ============================================================
# CNN
# ============================================================

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend = LogMelFrontEnd()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, waveform):
        x = self.frontend(waveform)
        x = self.features(x)
        logits = self.classifier(x).squeeze(1)
        return logits


# ============================================================
# MÉTRICAS
# ============================================================

def binary_metrics(labels, probs, threshold=0.5):
    labels = [int(x) for x in labels]
    preds = [
        1 if p >= threshold else 0
        for p in probs
    ]

    tp = sum(
        y == 1 and p == 1
        for y, p in zip(labels, preds)
    )
    tn = sum(
        y == 0 and p == 0
        for y, p in zip(labels, preds)
    )
    fp = sum(
        y == 0 and p == 1
        for y, p in zip(labels, preds)
    )
    fn = sum(
        y == 1 and p == 0
        for y, p in zip(labels, preds)
    )

    total = len(labels)

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_curve_manual(labels, probs):
    """
    Devuelve puntos (FPR, TPR, threshold).
    Sin sklearn.
    """
    pairs = sorted(
        zip(probs, labels),
        key=lambda x: x[0],
        reverse=True,
    )

    positives = sum(int(y) == 1 for y in labels)
    negatives = sum(int(y) == 0 for y in labels)

    if positives == 0 or negatives == 0:
        return []

    tp = 0
    fp = 0

    points = [(0.0, 0.0, float("inf"))]

    last_score = None

    for score, label in pairs:
        if last_score is not None and score != last_score:
            points.append((
                fp / negatives,
                tp / positives,
                last_score,
            ))

        if int(label) == 1:
            tp += 1
        else:
            fp += 1

        last_score = score

    points.append((
        fp / negatives,
        tp / positives,
        last_score if last_score is not None else 0.0,
    ))

    return points


def roc_auc_manual(labels, probs):
    points = roc_curve_manual(labels, probs)

    if len(points) < 2:
        return float("nan")

    # Ordenar por FPR
    points = sorted(points, key=lambda x: x[0])

    auc = 0.0

    for i in range(1, len(points)):
        x1, y1, _ = points[i - 1]
        x2, y2, _ = points[i]

        auc += (
            (x2 - x1)
            * (y1 + y2)
            / 2.0
        )

    return auc


def eer_manual(labels, probs):
    points = roc_curve_manual(labels, probs)

    if not points:
        return float("nan"), float("nan")

    best = None

    for fpr, tpr, threshold in points:
        fnr = 1.0 - tpr
        diff = abs(fpr - fnr)

        if best is None or diff < best[0]:
            best = (
                diff,
                (fpr + fnr) / 2.0,
                threshold,
            )

    return best[1], best[2]


# ============================================================
# DATALOADERS
# ============================================================

def make_loaders(rows):
    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row for row in rows
        if row["split"] == "val"
    ]

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    print("\nMuestras:")
    print(f"  train: {len(train_rows)}")
    print(f"  val  : {len(val_rows)}")
    print(f"  test : {len(test_rows)}")

    train_dataset = AudioDataset(
        train_rows,
        training=True,
    )

    val_dataset = AudioDataset(
        val_rows,
        training=False,
    )

    test_dataset = AudioDataset(
        test_rows,
        training=False,
    )

    # --------------------------------------------------------
    # WeightedRandomSampler:
    # compensa REAL / FAKE durante TRAIN sin eliminar fakes
    # --------------------------------------------------------

    train_labels = [
        int(row["label"])
        for row in train_rows
    ]

    counts = Counter(train_labels)

    if 0 not in counts or 1 not in counts:
        raise RuntimeError(
            f"Train necesita ambas clases. Counts: {dict(counts)}"
        )

    class_weight = {
        label: 1.0 / count
        for label, count in counts.items()
    }

    sample_weights = torch.tensor(
        [
            class_weight[label]
            for label in train_labels
        ],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nTrain original:")
    print(f"  REAL: {counts[0]}")
    print(f"  FAKE: {counts[1]}")
    print(
        "  Sampler: balance aproximado 50/50 "
        "durante entrenamiento."
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    labels_all = []
    probs_all = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            waveforms = batch["waveform"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(waveforms)

            loss = criterion(
                logits,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)

            bs = labels.size(0)

            total_loss += (
                loss.item() * bs
            )

            total_samples += bs

            labels_all.extend(
                labels.detach().cpu().tolist()
            )

            probs_all.extend(
                probs.detach().cpu().tolist()
            )

    avg_loss = (
        total_loss / total_samples
        if total_samples
        else 0.0
    )

    metrics = binary_metrics(
        labels_all,
        probs_all,
    )

    metrics["roc_auc"] = roc_auc_manual(
        labels_all,
        probs_all,
    )

    eer, eer_threshold = eer_manual(
        labels_all,
        probs_all,
    )

    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold
    metrics["loss"] = avg_loss

    return metrics


def evaluate_with_predictions(
    model,
    loader,
    device,
):
    model.eval()

    rows = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"]

            logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu()

            for i in range(len(labels)):
                label = int(labels[i].item())
                prob = float(probs[i].item())
                pred = 1 if prob >= 0.5 else 0

                rows.append({
                    "path": batch["path"][i],
                    "label": label,
                    "prob_fake": prob,
                    "prediction": pred,
                    "generator": batch["generator"][i],
                    "speaker_id": batch["speaker_id"][i],
                })

                labels_all.append(label)
                probs_all.append(prob)

    metrics = binary_metrics(
        labels_all,
        probs_all,
    )

    metrics["roc_auc"] = roc_auc_manual(
        labels_all,
        probs_all,
    )

    eer, eer_threshold = eer_manual(
        labels_all,
        probs_all,
    )

    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold

    return metrics, rows


# ============================================================
# GUARDADO
# ============================================================

def save_history(history):
    if not history:
        return

    with HISTORY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )

        writer.writeheader()
        writer.writerows(history)


def save_predictions(rows):
    if not rows:
        return

    with TEST_PREDICTIONS_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("BASELINE: LOG-MEL + CNN")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    rows = read_metadata()

    train_loader, val_loader, test_loader = make_loaders(
        rows
    )

    model = BaselineCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        print()
        print(
            f"================ EPOCH {epoch}/{EPOCHS} ================"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        scheduler.step(
            val_metrics["f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            "TRAIN | "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f} "
            f"auc={train_metrics['roc_auc']:.4f}"
        )

        print(
            "VAL   | "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"auc={val_metrics['roc_auc']:.4f} "
            f"eer={val_metrics['eer']:.4f}"
        )

        print(
            "VAL CM| "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} "
            f"TP={val_metrics['tp']}"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],

            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_eer": val_metrics["eer"],
        })

        save_history(history)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                    "config": {
                        "sample_rate": SAMPLE_RATE,
                        "clip_seconds": CLIP_SECONDS,
                        "n_mels": N_MELS,
                        "n_fft": N_FFT,
                        "hop_length": HOP_LENGTH,
                    },
                },
                BEST_MODEL,
            )

            print(
                f"BEST -> guardado: {BEST_MODEL}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Sin mejora:",
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print("Early stopping.")
                break

    # ========================================================
    # TEST FINAL
    # ========================================================

    checkpoint = torch.load(
        BEST_MODEL,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print("=" * 70)
    print(
        f"TEST usando mejor epoch: "
        f"{checkpoint['epoch']}"
    )
    print("=" * 70)

    test_metrics, predictions = evaluate_with_predictions(
        model,
        test_loader,
        device,
    )

    print(
        f"Accuracy : {test_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision: {test_metrics['precision']:.4f}"
    )
    print(
        f"Recall   : {test_metrics['recall']:.4f}"
    )
    print(
        f"F1       : {test_metrics['f1']:.4f}"
    )
    print(
        f"ROC-AUC  : {test_metrics['roc_auc']:.4f}"
    )
    print(
        f"EER      : {test_metrics['eer']:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )
    print(
        f"TN={test_metrics['tn']}  "
        f"FP={test_metrics['fp']}"
    )
    print(
        f"FN={test_metrics['fn']}  "
        f"TP={test_metrics['tp']}"
    )

    with TEST_METRICS_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_predictions(
        predictions
    )

    print()
    print("Resultados guardados en:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
