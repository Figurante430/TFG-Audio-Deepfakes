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

OUTPUT_ROOT = ROOT / "experiments" / "logo_logmel_cnn"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GENERATORS = [
    "confucius4",
    "cosyvoice",
    "knnvc",
    "openvoice",
    "qwen",
    "rvc",
    "voxcpm2",
]

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
NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METADATA
# ============================================================

def read_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"No encuentro:\n{METADATA_CSV}")

    with METADATA_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
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
        "utterance_id",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "Faltan columnas: " + ", ".join(sorted(missing))
        )

    return [
        row for row in rows
        if row["split"] in {"train", "val", "test"}
    ]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


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

        # [samples, channels] -> mono
        audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).float()

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sr,
                SAMPLE_RATE,
            ).squeeze(0)

        return waveform

    def _crop_or_pad(self, waveform):
        n = waveform.numel()

        if n > NUM_SAMPLES:
            max_start = n - NUM_SAMPLES

            if self.training:
                start = int(
                    torch.randint(
                        0,
                        max_start + 1,
                        (1,),
                    ).item()
                )
            else:
                start = max_start // 2

            waveform = waveform[start:start + NUM_SAMPLES]

        elif n < NUM_SAMPLES:
            missing = NUM_SAMPLES - n
            left = missing // 2
            right = missing - left
            waveform = F.pad(waveform, (left, right))

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
# MODELO
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
        x = self.mel(waveform)
        x = self.db(x)

        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        x = (x - mean) / std
        return x.unsqueeze(1)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend = LogMelFrontEnd()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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
        return self.classifier(x).squeeze(1)


# ============================================================
# MÉTRICAS
# ============================================================

def binary_metrics(labels, probs, threshold=0.5):
    labels = [int(x) for x in labels]
    preds = [1 if p >= threshold else 0 for p in probs]

    tp = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, preds))

    total = len(labels)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_curve_manual(labels, probs):
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
            points.append(
                (fp / negatives, tp / positives, last_score)
            )

        if int(label) == 1:
            tp += 1
        else:
            fp += 1

        last_score = score

    points.append(
        (
            fp / negatives,
            tp / positives,
            last_score if last_score is not None else 0.0,
        )
    )

    return points


def roc_auc_manual(labels, probs):
    points = roc_curve_manual(labels, probs)

    if len(points) < 2:
        return float("nan")

    points = sorted(points, key=lambda x: x[0])

    auc = 0.0

    for i in range(1, len(points)):
        x1, y1, _ = points[i - 1]
        x2, y2, _ = points[i]
        auc += (x2 - x1) * (y1 + y2) / 2.0

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


def find_best_f1_threshold(labels, probs):
    """
    El threshold se elige SOLO con validación.
    Nunca usamos test para calibrarlo.
    """
    candidates = sorted(set(float(p) for p in probs))

    if not candidates:
        return 0.5, 0.0

    # Añadimos extremos útiles
    candidates = [0.0] + candidates + [1.0]

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        metrics = binary_metrics(
            labels,
            probs,
            threshold=threshold,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# LOADERS
# ============================================================

def make_loader(rows, training):
    dataset = AudioDataset(
        rows,
        training=training,
    )

    if training:
        labels = [int(row["label"]) for row in rows]
        counts = Counter(labels)

        if 0 not in counts or 1 not in counts:
            raise RuntimeError(
                f"Train necesita real y fake. Counts={dict(counts)}"
            )

        class_weight = {
            label: 1.0 / count
            for label, count in counts.items()
        }

        weights = torch.tensor(
            [class_weight[label] for label in labels],
            dtype=torch.double,
        )

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# TRAIN / INFERENCE
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    model.train() if training else model.eval()

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
                optimizer.zero_grad(set_to_none=True)

            logits = model(waveforms)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            labels_all.extend(
                labels.detach().cpu().tolist()
            )
            probs_all.extend(
                probs.detach().cpu().tolist()
            )

    metrics = binary_metrics(
        labels_all,
        probs_all,
        threshold=0.5,
    )

    metrics["loss"] = (
        total_loss / total_samples
        if total_samples
        else 0.0
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

    return metrics, labels_all, probs_all


def predict(
    model,
    loader,
    device,
    threshold,
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

            logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu()

            for i in range(len(probs)):
                label = int(batch["label"][i].item())
                prob = float(probs[i].item())
                pred = 1 if prob >= threshold else 0

                rows.append({
                    "path": batch["path"][i],
                    "label": label,
                    "prediction": pred,
                    "prob_fake": prob,
                    "generator": batch["generator"][i],
                    "speaker_id": batch["speaker_id"][i],
                })

                labels_all.append(label)
                probs_all.append(prob)

    metrics = binary_metrics(
        labels_all,
        probs_all,
        threshold=threshold,
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
# CSV
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# UN FOLD LOGO
# ============================================================

def run_logo_fold(all_rows, held_out, device):
    print()
    print("=" * 78)
    print(f"LOGO: GENERADOR NO VISTO = {held_out}")
    print("=" * 78)

    fold_dir = OUTPUT_ROOT / f"holdout_{held_out}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = fold_dir / "best_model.pt"
    history_path = fold_dir / "history.csv"
    predictions_path = fold_dir / "test_predictions.csv"
    metrics_path = fold_dir / "metrics.json"

    # --------------------------------------------------------
    # TRAIN:
    #   split=train
    #   real + todos los fakes EXCEPTO held_out
    #
    # VAL:
    #   split=val
    #   real + todos los fakes EXCEPTO held_out
    #
    # TEST:
    #   split=test
    #   real + SOLO held_out
    #
    # Así el generador holdout no aparece jamás en train/val.
    # Y mantenemos la separación por utterance_id existente.
    # --------------------------------------------------------

    train_rows = [
        row for row in all_rows
        if row["split"] == "train"
        and (
            row["label"] == "0"
            or row["generator"] != held_out
        )
    ]

    val_rows = [
        row for row in all_rows
        if row["split"] == "val"
        and (
            row["label"] == "0"
            or row["generator"] != held_out
        )
    ]

    test_rows = [
        row for row in all_rows
        if row["split"] == "test"
        and (
            row["label"] == "0"
            or row["generator"] == held_out
        )
    ]

    # Comprobación estricta
    leaked_train = [
        row for row in train_rows
        if row["generator"] == held_out
    ]
    leaked_val = [
        row for row in val_rows
        if row["generator"] == held_out
    ]

    if leaked_train or leaked_val:
        raise RuntimeError(
            f"ERROR: {held_out} se ha colado en train/val."
        )

    print(f"Train: {len(train_rows)}")
    print(f"Val  : {len(val_rows)}")
    print(f"Test : {len(test_rows)}")

    print(
        "Train generators:",
        dict(
            sorted(
                Counter(
                    row["generator"]
                    for row in train_rows
                ).items()
            )
        )
    )

    print(
        "Test generators:",
        dict(
            sorted(
                Counter(
                    row["generator"]
                    for row in test_rows
                ).items()
            )
        )
    )

    train_loader = make_loader(
        train_rows,
        training=True,
    )

    val_loader = make_loader(
        val_rows,
        training=False,
    )

    test_loader = make_loader(
        test_rows,
        training=False,
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
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_metrics, val_labels, val_probs = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        scheduler.step(val_metrics["f1"])

        print(
            f"[{held_out}] epoch {epoch:02d} | "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_eer={val_metrics['eer']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
            "val_eer": val_metrics["eer"],
        })

        write_csv(history_path, history)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0

            best_threshold, best_threshold_f1 = (
                find_best_f1_threshold(
                    val_labels,
                    val_probs,
                )
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "held_out": held_out,
                    "val_f1": best_val_f1,
                    "val_threshold": best_threshold,
                    "val_threshold_f1": best_threshold_f1,
                },
                best_model_path,
            )

            print(
                f"  BEST | threshold_val={best_threshold:.4f}"
            )

        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print("  Early stopping.")
                break

    # --------------------------------------------------------
    # TEST FINAL
    # --------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    calibrated_threshold = float(
        checkpoint["val_threshold"]
    )

    # Resultado estándar threshold=0.5
    metrics_05, predictions_05 = predict(
        model,
        test_loader,
        device,
        threshold=0.5,
    )

    # Resultado threshold calibrado SOLO en validación
    metrics_cal, predictions_cal = predict(
        model,
        test_loader,
        device,
        threshold=calibrated_threshold,
    )

    # Guardamos predicciones calibradas
    write_csv(
        predictions_path,
        predictions_cal,
    )

    fold_metrics = {
        "held_out_generator": held_out,
        "best_epoch": int(checkpoint["epoch"]),
        "val_f1": float(checkpoint["val_f1"]),
        "calibrated_threshold": calibrated_threshold,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),

        "threshold_0.5": metrics_05,
        "threshold_calibrated": metrics_cal,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fold_metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"TEST LOGO [{held_out}]")
    print(
        "  threshold=0.5 | "
        f"acc={metrics_05['accuracy']:.4f} "
        f"prec={metrics_05['precision']:.4f} "
        f"rec={metrics_05['recall']:.4f} "
        f"spec={metrics_05['specificity']:.4f} "
        f"f1={metrics_05['f1']:.4f} "
        f"auc={metrics_05['roc_auc']:.4f} "
        f"eer={metrics_05['eer']:.4f}"
    )

    print(
        f"  threshold={calibrated_threshold:.4f} | "
        f"acc={metrics_cal['accuracy']:.4f} "
        f"prec={metrics_cal['precision']:.4f} "
        f"rec={metrics_cal['recall']:.4f} "
        f"spec={metrics_cal['specificity']:.4f} "
        f"f1={metrics_cal['f1']:.4f} "
        f"auc={metrics_cal['roc_auc']:.4f} "
        f"eer={metrics_cal['eer']:.4f}"
    )

    return {
        "held_out_generator": held_out,
        "best_epoch": int(checkpoint["epoch"]),
        "threshold_val": calibrated_threshold,

        "acc_05": metrics_05["accuracy"],
        "precision_05": metrics_05["precision"],
        "recall_05": metrics_05["recall"],
        "specificity_05": metrics_05["specificity"],
        "f1_05": metrics_05["f1"],
        "auc_05": metrics_05["roc_auc"],
        "eer_05": metrics_05["eer"],

        "acc_cal": metrics_cal["accuracy"],
        "precision_cal": metrics_cal["precision"],
        "recall_cal": metrics_cal["recall"],
        "specificity_cal": metrics_cal["specificity"],
        "f1_cal": metrics_cal["f1"],
        "auc_cal": metrics_cal["roc_auc"],
        "eer_cal": metrics_cal["eer"],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("LEAVE-ONE-GENERATOR-OUT (LOGO)")
    print("=" * 78)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    all_rows = read_metadata()

    available_generators = {
        row["generator"]
        for row in all_rows
        if row["label"] == "1"
    }

    missing = [
        gen
        for gen in GENERATORS
        if gen not in available_generators
    ]

    if missing:
        raise RuntimeError(
            "Faltan generadores en metadata: "
            + ", ".join(missing)
        )

    summary_rows = []

    for fold_idx, held_out in enumerate(
        GENERATORS,
        1
    ):
        # Reiniciar seeds en cada fold para hacerlos comparables
        seed_everything(SEED + fold_idx)

        result = run_logo_fold(
            all_rows,
            held_out,
            device,
        )

        summary_rows.append(result)

        write_csv(
            OUTPUT_ROOT / "logo_summary.csv",
            summary_rows,
        )

        # Liberar VRAM entre folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("RESUMEN LOGO")
    print("=" * 78)

    for row in summary_rows:
        print(
            f"{row['held_out_generator']:12s} | "
            f"F1={row['f1_cal']:.4f} "
            f"AUC={row['auc_cal']:.4f} "
            f"EER={row['eer_cal']:.4f} "
            f"Recall={row['recall_cal']:.4f} "
            f"Spec={row['specificity_cal']:.4f}"
        )

    print()
    print(
        "Resultados:",
        OUTPUT_ROOT / "logo_summary.csv"
    )


if __name__ == "__main__":
    main()
