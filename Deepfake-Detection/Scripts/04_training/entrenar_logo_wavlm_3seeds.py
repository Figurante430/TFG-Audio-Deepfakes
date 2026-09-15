from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
METADATA_CSV = ROOT / "data_normalized" / "metadata_normalized.csv"

OUTPUT_ROOT = ROOT / "experiments" / "logo_wavlm_3seeds"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/wavlm-base-plus"

GENERATORS = [
    "confucius4",
    "cosyvoice",
    "knnvc",
    "openvoice",
    "qwen",
    "rvc",
    "voxcpm2",
]

# Las MISMAS tres semillas para TODOS los generadores.
SEEDS = [42, 123, 2026]

SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Configuración conservadora para ~8 GB VRAM
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
NUM_WORKERS = 0

# Misma estrategia usada en el WavLM LOGO anterior
HEAD_EPOCHS = 2
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 8
FINETUNE_LAST_N_LAYERS = 4
BACKBONE_LR = 1e-5
HEAD_FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3
USE_AMP = True


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
        "speaker_id",
        "utterance_id",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "Faltan columnas en metadata: "
            + ", ".join(sorted(missing))
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

        if sr != SAMPLE_RATE:
            raise RuntimeError(
                f"{path.name}: sample rate={sr}; "
                f"esperaba {SAMPLE_RATE}."
            )

        audio = audio.mean(axis=1)
        return torch.from_numpy(audio).float()

    def _crop_or_pad(self, waveform):
        n = waveform.numel()

        if n >= NUM_SAMPLES:
            max_start = n - NUM_SAMPLES

            if self.training and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            waveform = waveform[
                start:start + NUM_SAMPLES
            ]

            attention_mask = torch.ones(
                NUM_SAMPLES,
                dtype=torch.long,
            )

        else:
            missing = NUM_SAMPLES - n
            left = missing // 2
            right = missing - left

            waveform = F.pad(
                waveform,
                (left, right),
            )

            attention_mask = torch.zeros(
                NUM_SAMPLES,
                dtype=torch.long,
            )

            attention_mask[left:left + n] = 1

        return waveform, attention_mask

    def __getitem__(self, index):
        row = self.rows[index]

        waveform = self._load_audio(
            resolve_path(row["path"])
        )

        waveform, attention_mask = self._crop_or_pad(
            waveform
        )

        return {
            "input_values": waveform,
            "attention_mask": attention_mask,
            "label": torch.tensor(
                float(row["label"]),
                dtype=torch.float32,
            ),
            "path": row["path"],
            "generator": row["generator"],
            "speaker_id": row["speaker_id"],
        }


# ============================================================
# MODELO
# ============================================================

class WavLMDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )

    def masked_mean_pool(
        self,
        hidden_states,
        attention_mask,
    ):
        if hasattr(
            self.backbone,
            "_get_feature_vector_attention_mask",
        ):
            feature_mask = (
                self.backbone
                ._get_feature_vector_attention_mask(
                    hidden_states.shape[1],
                    attention_mask,
                )
            )

            mask = (
                feature_mask
                .unsqueeze(-1)
                .to(hidden_states.dtype)
            )

            summed = (
                hidden_states * mask
            ).sum(dim=1)

            denom = (
                mask.sum(dim=1)
                .clamp_min(1.0)
            )

            return summed / denom

        return hidden_states.mean(dim=1)

    def forward(
        self,
        input_values,
        attention_mask,
    ):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        pooled = self.masked_mean_pool(
            hidden,
            attention_mask,
        )

        return (
            self.classifier(pooled)
            .squeeze(1)
        )


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_wavlm_layers(
    model,
    n_layers,
):
    for p in model.backbone.parameters():
        p.requires_grad = False

    layers = model.backbone.encoder.layers

    n_layers = min(
        n_layers,
        len(layers),
    )

    for layer in layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(
        model.backbone.encoder,
        "layer_norm",
    ):
        for p in (
            model.backbone.encoder
            .layer_norm
            .parameters()
        ):
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True


def count_trainable(model):
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    return trainable, total


# ============================================================
# MÉTRICAS
# ============================================================

def binary_metrics(
    labels,
    probs,
    threshold=0.5,
):
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
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
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

    positives = sum(
        int(y) == 1
        for y in labels
    )

    negatives = sum(
        int(y) == 0
        for y in labels
    )

    if positives == 0 or negatives == 0:
        return []

    tp = 0
    fp = 0

    points = [
        (0.0, 0.0, float("inf"))
    ]

    last_score = None

    for score, label in pairs:
        if (
            last_score is not None
            and score != last_score
        ):
            points.append(
                (
                    fp / negatives,
                    tp / positives,
                    last_score,
                )
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
            last_score
            if last_score is not None
            else 0.0,
        )
    )

    return points


def roc_auc_manual(labels, probs):
    points = roc_curve_manual(
        labels,
        probs,
    )

    if len(points) < 2:
        return float("nan")

    points = sorted(
        points,
        key=lambda x: x[0],
    )

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
    points = roc_curve_manual(
        labels,
        probs,
    )

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


def best_f1_threshold(labels, probs):
    candidates = sorted(
        set(float(p) for p in probs)
    )

    if not candidates:
        return 0.5, 0.0

    candidates = (
        [0.0]
        + candidates
        + [1.0]
    )

    best_t = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        metrics = binary_metrics(
            labels,
            probs,
            threshold,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = threshold

    return best_t, best_f1


# ============================================================
# LOADERS
# ============================================================

def make_loader(rows, training):
    dataset = AudioDataset(
        rows,
        training=training,
    )

    if training:
        labels = [
            int(row["label"])
            for row in rows
        ]

        counts = Counter(labels)

        if 0 not in counts or 1 not in counts:
            raise RuntimeError(
                f"Train necesita ambas clases: {dict(counts)}"
            )

        class_weight = {
            label: 1.0 / count
            for label, count in counts.items()
        }

        weights = torch.tensor(
            [
                class_weight[label]
                for label in labels
            ],
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
# OPTIMIZADORES
# ============================================================

def make_head_optimizer(model):
    return torch.optim.AdamW(
        model.classifier.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )


def make_finetune_optimizer(model):
    backbone_params = [
        p
        for p in model.backbone.parameters()
        if p.requires_grad
    ]

    head_params = [
        p
        for p in model.classifier.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_FINETUNE_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# AMP
# ============================================================

def make_scaler(device):
    enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def autocast_context(device):
    enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_samples = 0

    labels_all = []
    probs_all = []

    num_batches = len(loader)

    for batch_idx, batch in enumerate(
        loader,
        1,
    ):
        inputs = batch["input_values"].to(
            device,
            non_blocking=True,
        )

        mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with autocast_context(device):
            logits = model(
                inputs,
                mask,
            )

            raw_loss = criterion(
                logits,
                labels,
            )

            loss = (
                raw_loss
                / GRAD_ACCUM_STEPS
            )

        scaler.scale(loss).backward()

        should_step = (
            batch_idx % GRAD_ACCUM_STEPS == 0
            or batch_idx == num_batches
        )

        if should_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                    and p.grad is not None
                ],
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        probs = (
            torch.sigmoid(logits)
            .detach()
            .cpu()
        )

        bs = labels.size(0)

        total_loss += (
            raw_loss.item() * bs
        )

        total_samples += bs

        labels_all.extend(
            labels.detach().cpu().tolist()
        )

        probs_all.extend(
            probs.tolist()
        )

    metrics = binary_metrics(
        labels_all,
        probs_all,
        0.5,
    )

    metrics["loss"] = (
        total_loss / total_samples
    )

    metrics["roc_auc"] = roc_auc_manual(
        labels_all,
        probs_all,
    )

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
    return_predictions=False,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    labels_all = []
    probs_all = []

    prediction_rows = []

    for batch in loader:
        inputs = batch["input_values"].to(
            device,
            non_blocking=True,
        )

        mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with autocast_context(device):
            logits = model(
                inputs,
                mask,
            )

            loss = criterion(
                logits,
                labels,
            )

        probs = torch.sigmoid(logits)

        bs = labels.size(0)

        total_loss += loss.item() * bs
        total_samples += bs

        batch_labels = (
            labels.detach()
            .cpu()
            .tolist()
        )

        batch_probs = (
            probs.detach()
            .cpu()
            .tolist()
        )

        labels_all.extend(batch_labels)
        probs_all.extend(batch_probs)

        if return_predictions:
            for i in range(bs):
                prob = float(batch_probs[i])
                label = int(batch_labels[i])

                prediction_rows.append({
                    "path": batch["path"][i],
                    "label": label,
                    "prob_fake": prob,
                    "prediction": (
                        1
                        if prob >= threshold
                        else 0
                    ),
                    "generator": batch["generator"][i],
                    "speaker_id": batch["speaker_id"][i],
                })

    metrics = binary_metrics(
        labels_all,
        probs_all,
        threshold,
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

    return (
        metrics,
        labels_all,
        probs_all,
        prediction_rows,
    )


# ============================================================
# CSV
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
# UN RUN: GENERADOR + SEMILLA
# ============================================================

def run_one(
    all_rows,
    held_out,
    seed,
    device,
    force=False,
):
    seed_everything(seed)

    run_dir = (
        OUTPUT_ROOT
        / f"holdout_{held_out}"
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        run_dir / "best_model.pt"
    )

    history_path = (
        run_dir / "history.csv"
    )

    metrics_path = (
        run_dir / "metrics.json"
    )

    predictions_path = (
        run_dir / "test_predictions.csv"
    )

    # --------------------------------------------------------
    # REANUDACIÓN:
    # si metrics.json existe, reutilizar el resultado.
    # --------------------------------------------------------

    if (
        metrics_path.exists()
        and not force
    ):
        print()
        print(
            f"[SKIP] {held_out} seed={seed}: "
            "ya existe metrics.json"
        )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            saved = json.load(f)

        return saved["summary_row"]

    print()
    print("=" * 86)
    print(
        f"WavLM LOGO | holdout={held_out} | seed={seed}"
    )
    print("=" * 86)

    train_rows = [
        row
        for row in all_rows
        if row["split"] == "train"
        and (
            row["label"] == "0"
            or row["generator"] != held_out
        )
    ]

    val_rows = [
        row
        for row in all_rows
        if row["split"] == "val"
        and (
            row["label"] == "0"
            or row["generator"] != held_out
        )
    ]

    test_rows = [
        row
        for row in all_rows
        if row["split"] == "test"
        and (
            row["label"] == "0"
            or row["generator"] == held_out
        )
    ]

    if any(
        row["generator"] == held_out
        for row in train_rows + val_rows
    ):
        raise RuntimeError(
            f"{held_out} se ha colado en train/val."
        )

    print(f"Train: {len(train_rows)}")
    print(f"Val  : {len(val_rows)}")
    print(f"Test : {len(test_rows)}")

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

    print(
        f"Cargando {MODEL_NAME}..."
    )

    model = WavLMDetector().to(device)

    criterion = nn.BCEWithLogitsLoss()

    history = []

    best_val_auc = -1.0
    best_epoch = 0

    global_epoch = 0

    # --------------------------------------------------------
    # FASE 1: HEAD
    # --------------------------------------------------------

    freeze_backbone(model)

    trainable, total = count_trainable(
        model
    )

    print(
        f"Fase HEAD | trainable="
        f"{trainable:,}/{total:,}"
    )

    optimizer = make_head_optimizer(
        model
    )

    scaler = make_scaler(device)

    for _ in range(HEAD_EPOCHS):
        global_epoch += 1

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        (
            val_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
        )

        print(
            f"[{held_out} seed={seed}] "
            f"HEAD epoch {global_epoch:02d} | "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_eer={val_metrics['eer']:.4f}"
        )

        history.append({
            "seed": seed,
            "epoch": global_epoch,
            "stage": "head",
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
            "val_eer": val_metrics["eer"],
        })

        write_csv(
            history_path,
            history,
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = global_epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": global_epoch,
                    "stage": "head",
                    "held_out": held_out,
                    "seed": seed,
                    "val_auc": best_val_auc,
                },
                best_model_path,
            )

            print("  BEST por val_auc")

    # --------------------------------------------------------
    # FASE 2: FINETUNE ÚLTIMAS 4 CAPAS
    # --------------------------------------------------------

    unfreeze_last_wavlm_layers(
        model,
        FINETUNE_LAST_N_LAYERS,
    )

    trainable, total = count_trainable(
        model
    )

    print(
        f"Fase FINETUNE({FINETUNE_LAST_N_LAYERS}) | "
        f"trainable={trainable:,}/{total:,}"
    )

    optimizer = make_finetune_optimizer(
        model
    )

    scaler = make_scaler(device)

    epochs_without_improvement = 0

    for _ in range(FINETUNE_EPOCHS):
        global_epoch += 1

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        (
            val_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
        )

        print(
            f"[{held_out} seed={seed}] "
            f"FT epoch {global_epoch:02d} | "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_eer={val_metrics['eer']:.4f}"
        )

        history.append({
            "seed": seed,
            "epoch": global_epoch,
            "stage": "finetune",
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
            "val_eer": val_metrics["eer"],
        })

        write_csv(
            history_path,
            history,
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = global_epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": global_epoch,
                    "stage": "finetune",
                    "held_out": held_out,
                    "seed": seed,
                    "val_auc": best_val_auc,
                },
                best_model_path,
            )

            print("  BEST por val_auc")

        else:
            epochs_without_improvement += 1

            print(
                f"  Sin mejora AUC: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print("  Early stopping.")
                break

    # --------------------------------------------------------
    # MEJOR CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # THRESHOLD SOLO EN VALIDACIÓN
    # --------------------------------------------------------

    (
        _,
        val_labels,
        val_probs,
        _,
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device,
        threshold=0.5,
    )

    calibrated_threshold, val_best_f1 = (
        best_f1_threshold(
            val_labels,
            val_probs,
        )
    )

    # --------------------------------------------------------
    # TEST threshold=0.5
    # --------------------------------------------------------

    (
        metrics_05,
        _,
        _,
        _,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=0.5,
    )

    # --------------------------------------------------------
    # TEST threshold calibrado
    # --------------------------------------------------------

    (
        metrics_cal,
        _,
        _,
        predictions,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=calibrated_threshold,
        return_predictions=True,
    )

    for row in predictions:
        row["prediction_calibrated"] = row.pop(
            "prediction"
        )

        row["prediction_05"] = (
            1
            if row["prob_fake"] >= 0.5
            else 0
        )

        row["threshold_calibrated"] = (
            calibrated_threshold
        )

        row["seed"] = seed

    write_csv(
        predictions_path,
        predictions,
    )

    summary_row = {
        "held_out_generator": held_out,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_stage": checkpoint["stage"],
        "best_val_auc": float(best_val_auc),
        "threshold_val": float(calibrated_threshold),

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

        "tp_cal": metrics_cal["tp"],
        "tn_cal": metrics_cal["tn"],
        "fp_cal": metrics_cal["fp"],
        "fn_cal": metrics_cal["fn"],
    }

    complete_metrics = {
        "model": MODEL_NAME,
        "held_out_generator": held_out,
        "seed": seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "best_epoch": int(checkpoint["epoch"]),
        "best_stage": checkpoint["stage"],
        "best_val_auc": float(best_val_auc),
        "calibrated_threshold": float(
            calibrated_threshold
        ),
        "val_best_f1_at_calibrated_threshold": float(
            val_best_f1
        ),
        "threshold_0.5": metrics_05,
        "threshold_calibrated": metrics_cal,
        "summary_row": summary_row,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            complete_metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"TEST [{held_out}] seed={seed}"
    )

    print(
        f"  threshold={calibrated_threshold:.6f} | "
        f"acc={metrics_cal['accuracy']:.4f} "
        f"prec={metrics_cal['precision']:.4f} "
        f"rec={metrics_cal['recall']:.4f} "
        f"spec={metrics_cal['specificity']:.4f} "
        f"f1={metrics_cal['f1']:.4f} "
        f"auc={metrics_cal['roc_auc']:.4f} "
        f"eer={metrics_cal['eer']:.4f}"
    )

    # --------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------

    del model
    del optimizer
    del train_loader
    del val_loader
    del test_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_row


# ============================================================
# AGREGACIÓN MEDIA ± DESVIACIÓN ESTÁNDAR
# ============================================================

def mean_std(values):
    values = [
        float(v)
        for v in values
    ]

    mean = statistics.mean(values)

    # Desviación estándar muestral
    std = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return mean, std


def aggregate_results(run_rows):
    aggregate_rows = []

    for generator in GENERATORS:
        group = [
            row
            for row in run_rows
            if row["held_out_generator"]
            == generator
        ]

        if not group:
            continue

        aggregate = {
            "held_out_generator": generator,
            "n_seeds": len(group),
            "seeds": ",".join(
                str(row["seed"])
                for row in group
            ),
        }

        metrics = [
            "acc_05",
            "precision_05",
            "recall_05",
            "specificity_05",
            "f1_05",
            "auc_05",
            "eer_05",
            "acc_cal",
            "precision_cal",
            "recall_cal",
            "specificity_cal",
            "f1_cal",
            "auc_cal",
            "eer_cal",
            "threshold_val",
        ]

        for metric in metrics:
            mean, std = mean_std(
                [
                    row[metric]
                    for row in group
                ]
            )

            aggregate[
                f"{metric}_mean"
            ] = mean

            aggregate[
                f"{metric}_std"
            ] = std

        aggregate_rows.append(
            aggregate
        )

    return aggregate_rows


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "WavLM Leave-One-Generator-Out "
            "repetido con 3 semillas."
        )
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--holdout",
        choices=GENERATORS,
        help=(
            "Ejecuta las 3 semillas solo para "
            "un generador."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Ejecuta 7 generadores x 3 semillas = 21 runs."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Repite también runs que ya tengan metrics.json."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 86)
    print("WavLM LOGO - 3 SEMILLAS")
    print("=" * 86)
    print(f"Modelo   : {MODEL_NAME}")
    print(f"Seeds    : {SEEDS}")
    print(f"Device   : {device}")

    if torch.cuda.is_available():
        print(
            "GPU      :",
            torch.cuda.get_device_name(0),
        )

    all_rows = read_metadata()

    available_generators = {
        row["generator"]
        for row in all_rows
        if row["label"] == "1"
    }

    missing = [
        generator
        for generator in GENERATORS
        if generator not in available_generators
    ]

    if missing:
        raise RuntimeError(
            "Faltan generadores en metadata: "
            + ", ".join(missing)
        )

    if args.all:
        folds = GENERATORS

    elif args.holdout:
        folds = [args.holdout]

    else:
        # Por defecto probar primero RVC con las 3 seeds
        folds = ["rvc"]

        print()
        print(
            "No se indicó --all ni --holdout. "
            "Se ejecutarán las 3 semillas de RVC."
        )

    run_rows = []

    for held_out in folds:
        for seed in SEEDS:
            result = run_one(
                all_rows,
                held_out,
                seed,
                device,
                force=args.force,
            )

            run_rows.append(
                result
            )

            # Guardar progreso tras cada run
            all_runs_path = (
                OUTPUT_ROOT
                / "summary_all_runs.csv"
            )

            # Mezclar con runs previos existentes
            previous = []

            if all_runs_path.exists():
                try:
                    with all_runs_path.open(
                        "r",
                        encoding="utf-8-sig",
                        newline="",
                    ) as f:
                        previous = list(
                            csv.DictReader(f)
                        )
                except Exception:
                    previous = []

            by_key = {
                (
                    row["held_out_generator"],
                    str(row["seed"]),
                ): row
                for row in previous
            }

            for row in run_rows:
                by_key[
                    (
                        row["held_out_generator"],
                        str(row["seed"]),
                    )
                ] = row

            merged_runs = list(
                by_key.values()
            )

            write_csv(
                all_runs_path,
                merged_runs,
            )

            aggregate = aggregate_results(
                merged_runs
            )

            write_csv(
                OUTPUT_ROOT
                / "summary_mean_std.csv",
                aggregate,
            )

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    all_runs_path = (
        OUTPUT_ROOT
        / "summary_all_runs.csv"
    )

    with all_runs_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        all_run_rows = list(
            csv.DictReader(f)
        )

    aggregate_rows = aggregate_results(
        all_run_rows
    )

    write_csv(
        OUTPUT_ROOT
        / "summary_mean_std.csv",
        aggregate_rows,
    )

    print()
    print("=" * 86)
    print("RESUMEN MEDIA ± STD (threshold calibrado)")
    print("=" * 86)

    for row in aggregate_rows:
        generator = row[
            "held_out_generator"
        ]

        print(
            f"{generator:12s} | "
            f"F1={float(row['f1_cal_mean']):.4f}"
            f" ± {float(row['f1_cal_std']):.4f} | "
            f"AUC={float(row['auc_cal_mean']):.4f}"
            f" ± {float(row['auc_cal_std']):.4f} | "
            f"EER={float(row['eer_cal_mean']):.4f}"
            f" ± {float(row['eer_cal_std']):.4f} | "
            f"Recall={float(row['recall_cal_mean']):.4f}"
            f" ± {float(row['recall_cal_std']):.4f} | "
            f"Spec={float(row['specificity_cal_mean']):.4f}"
            f" ± {float(row['specificity_cal_std']):.4f}"
        )

    print()
    print(
        "Runs individuales:",
        OUTPUT_ROOT / "summary_all_runs.csv"
    )

    print(
        "Media y desviación:",
        OUTPUT_ROOT / "summary_mean_std.csv"
    )


if __name__ == "__main__":
    main()
