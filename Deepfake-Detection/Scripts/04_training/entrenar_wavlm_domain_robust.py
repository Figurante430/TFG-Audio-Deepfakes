from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import WavLMModel


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

ROOT_DEFAULT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
MODEL_NAME = "microsoft/wavlm-base-plus"

TARGET_SR = 16000
SECONDS = 4
MAX_SAMPLES = TARGET_SR * SECONDS

SEEDS = [42, 123, 2026]

BATCH_SIZE = 2
GRAD_ACCUM = 8

HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3

HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-2

LAST_N_LAYERS = 4

# ============================================================
# AUGMENTATION ROBUSTA
# Se aplica SOLO a TRAIN y de forma idéntica a real/fake.
# ============================================================

GAIN_PROB = 0.50
GAIN_DB_MIN = -6.0
GAIN_DB_MAX = +6.0

NOISE_PROB = 0.50
NOISE_SNR_DB_MIN = 18.0
NOISE_SNR_DB_MAX = 40.0

FLOOR_PROB = 0.55
FLOOR_GATE_PROB = 0.50
GATE_DBFS_MIN = -55.0
GATE_DBFS_MAX = -35.0
FLOOR_NOISE_DBFS_MIN = -55.0
FLOOR_NOISE_DBFS_MAX = -38.0

LOWPASS_PROB = 0.25
LOWPASS_HZ_MIN = 3200.0
LOWPASS_HZ_MAX = 7000.0
LOWPASS_TAPS = 257

REVERB_PROB = 0.25
REVERB_DELAY_MS_MIN = 15.0
REVERB_DELAY_MS_MAX = 70.0
REVERB_GAIN_MIN = 0.08
REVERB_GAIN_MAX = 0.30


# ============================================================
# UTILIDADES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_metadata(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_audio_path(root: Path, value: str) -> Path:
    p = Path(value)

    if p.is_absolute():
        return p

    return root / p


def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def rms(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) == 0:
        return 0.0

    return float(np.sqrt(np.mean(x * x)))


# ============================================================
# AUGMENTATIONS
# ============================================================

def random_gain(x):
    gain_db = random.uniform(
        GAIN_DB_MIN,
        GAIN_DB_MAX,
    )

    y = x * db_to_amp(gain_db)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.995:
        y = y * (0.995 / peak)

    return y.astype(np.float32)


def add_noise_snr(x):
    x = np.asarray(x, dtype=np.float32)

    signal_rms = rms(x)

    if signal_rms <= 1e-8:
        return x.copy()

    snr_db = random.uniform(
        NOISE_SNR_DB_MIN,
        NOISE_SNR_DB_MAX,
    )

    noise_rms = signal_rms / db_to_amp(snr_db)

    noise = np.random.normal(
        loc=0.0,
        scale=noise_rms,
        size=len(x),
    ).astype(np.float32)

    y = x + noise

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.995:
        y = y * (0.995 / peak)

    return y.astype(np.float32)


def random_floor_transform(x):
    """
    Rompe la correlación entre clase y near-zero/noise-floor.

    De forma aleatoria:
      A) noise gate: pequeños valores -> 0
      B) añade un noise floor suave a TODO el segmento válido

    Se aplica igual a real y fake.
    """
    x = np.asarray(x, dtype=np.float32).copy()

    if random.random() < FLOOR_GATE_PROB:
        threshold_db = random.uniform(
            GATE_DBFS_MIN,
            GATE_DBFS_MAX,
        )

        threshold = db_to_amp(threshold_db)

        x[np.abs(x) < threshold] = 0.0

        return x

    floor_db = random.uniform(
        FLOOR_NOISE_DBFS_MIN,
        FLOOR_NOISE_DBFS_MAX,
    )

    floor_amp = db_to_amp(floor_db)

    noise = np.random.normal(
        0.0,
        floor_amp,
        size=len(x),
    ).astype(np.float32)

    y = x + noise

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.995:
        y = y * (0.995 / peak)

    return y.astype(np.float32)


def fir_lowpass(x, cutoff_hz):
    x = np.asarray(x, dtype=np.float32)

    taps = LOWPASS_TAPS

    if taps % 2 == 0:
        taps += 1

    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / TARGET_SR

    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.hamming(taps)
    h /= np.sum(h)

    y = np.convolve(
        x.astype(np.float64),
        h.astype(np.float64),
        mode="same",
    )

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.995:
        y = y * (0.995 / peak)

    return y.astype(np.float32)


def simple_reverb(x):
    """
    Eco corto aleatorio, deliberadamente sencillo.
    Evita dependencias scipy/librosa.
    """
    x = np.asarray(x, dtype=np.float32)

    delay_ms = random.uniform(
        REVERB_DELAY_MS_MIN,
        REVERB_DELAY_MS_MAX,
    )

    gain = random.uniform(
        REVERB_GAIN_MIN,
        REVERB_GAIN_MAX,
    )

    delay = max(
        1,
        int(TARGET_SR * delay_ms / 1000.0),
    )

    y = x.astype(np.float64).copy()

    if len(x) > delay:
        y[delay:] += gain * x[:-delay]

    if len(x) > 2 * delay:
        y[2 * delay:] += (gain ** 2) * x[:-2 * delay]

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0.995:
        y = y * (0.995 / peak)

    return y.astype(np.float32)


def robust_augment(x):
    y = np.asarray(x, dtype=np.float32).copy()

    if random.random() < GAIN_PROB:
        y = random_gain(y)

    if random.random() < NOISE_PROB:
        y = add_noise_snr(y)

    if random.random() < FLOOR_PROB:
        y = random_floor_transform(y)

    if random.random() < LOWPASS_PROB:
        cutoff = random.uniform(
            LOWPASS_HZ_MIN,
            LOWPASS_HZ_MAX,
        )
        y = fir_lowpass(
            y,
            cutoff,
        )

    if random.random() < REVERB_PROB:
        y = simple_reverb(y)

    return y.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):
    def __init__(
        self,
        rows,
        root: Path,
        split: str,
        mode: str,
    ):
        self.rows = list(rows)
        self.root = root
        self.split = split
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def _crop_or_keep(self, x):
        if len(x) > MAX_SAMPLES:
            if self.split == "train":
                max_start = len(x) - MAX_SAMPLES
                start = random.randint(
                    0,
                    max_start,
                )
            else:
                start = (
                    len(x) - MAX_SAMPLES
                ) // 2

            x = x[
                start:start + MAX_SAMPLES
            ]

        return np.asarray(
            x,
            dtype=np.float32,
        )

    def __getitem__(self, idx):
        row = self.rows[idx]

        path = resolve_audio_path(
            self.root,
            row["path"],
        )

        x, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        if x.ndim == 2:
            x = x.mean(axis=1)

        if sr != TARGET_SR:
            raise RuntimeError(
                f"{path} está a {sr} Hz; esperaba {TARGET_SR}."
            )

        x = self._crop_or_keep(x)

        # La augmentation solo afecta al tramo de audio REAL,
        # antes del zero-padding.
        if (
            self.split == "train"
            and self.mode == "robust"
        ):
            x = robust_augment(x)

        valid_len = min(
            len(x),
            MAX_SAMPLES,
        )

        if len(x) < MAX_SAMPLES:
            x = np.pad(
                x,
                (0, MAX_SAMPLES - len(x)),
                mode="constant",
            )

        mask = np.zeros(
            MAX_SAMPLES,
            dtype=np.int64,
        )
        mask[:valid_len] = 1

        label = float(
            int(row["label"])
        )

        return {
            "input_values": torch.from_numpy(
                x.astype(np.float32)
            ),
            "attention_mask": torch.from_numpy(
                mask
            ),
            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),
            "path": str(path),
            "generator": row.get(
                "generator",
                "",
            ),
            "speaker_id": row.get(
                "speaker_id",
                "",
            ),
        }


# ============================================================
# MODELO
# ============================================================

class WavLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(
            MODEL_NAME
        )

        hidden = self.wavlm.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        input_values,
        attention_mask,
    ):
        out = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        h = out.last_hidden_state

        feat_mask = (
            self.wavlm
            ._get_feature_vector_attention_mask(
                h.shape[1],
                attention_mask,
            )
            .to(h.device)
        )

        feat_mask_f = (
            feat_mask
            .unsqueeze(-1)
            .to(h.dtype)
        )

        pooled = (
            h * feat_mask_f
        ).sum(dim=1)

        denom = (
            feat_mask_f
            .sum(dim=1)
            .clamp_min(1.0)
        )

        pooled = pooled / denom

        return self.head(
            pooled
        ).squeeze(-1)


def freeze_backbone(model):
    for p in model.wavlm.parameters():
        p.requires_grad = False

    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_last_layers(
    model,
    n_layers=LAST_N_LAYERS,
):
    for p in model.wavlm.parameters():
        p.requires_grad = False

    layers = model.wavlm.encoder.layers

    for layer in layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
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
# MÉTRICAS SIN SKLEARN/PANDAS
# ============================================================

def confusion_metrics(
    labels,
    probs,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    pred = (
        probs >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (pred == 1)
            & (labels == 1)
        )
    )

    tn = int(
        np.sum(
            (pred == 0)
            & (labels == 0)
        )
    )

    fp = int(
        np.sum(
            (pred == 1)
            & (labels == 0)
        )
    )

    fn = int(
        np.sum(
            (pred == 0)
            & (labels == 1)
        )
    )

    total = len(labels)

    accuracy = (
        (tp + tn) / total
        if total else 0.0
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

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
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


def rankdata(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    i = 0

    while i < len(values):
        j = i + 1

        while (
            j < len(values)
            and values[order[j]]
            == values[order[i]]
        ):
            j += 1

        avg_rank = (
            (i + j - 1) / 2.0
            + 1.0
        )

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    return ranks


def roc_auc_score_np(
    labels,
    probs,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    pos = labels == 1
    neg = labels == 0

    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(probs)

    sum_pos_ranks = float(
        ranks[pos].sum()
    )

    auc = (
        sum_pos_ranks
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def eer_np(
    labels,
    probs,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    thresholds = np.unique(
        np.concatenate(
            (
                np.array(
                    [-1e-9, 1.0 + 1e-9]
                ),
                probs,
            )
        )
    )

    best = None

    for t in thresholds:
        m = confusion_metrics(
            labels,
            probs,
            t,
        )

        fpr = (
            m["fp"]
            / (m["fp"] + m["tn"])
            if (m["fp"] + m["tn"])
            else 0.0
        )

        fnr = (
            m["fn"]
            / (m["fn"] + m["tp"])
            if (m["fn"] + m["tp"])
            else 0.0
        )

        diff = abs(
            fpr - fnr
        )

        eer = (
            fpr + fnr
        ) / 2.0

        item = (
            diff,
            eer,
            float(t),
        )

        if (
            best is None
            or item[0] < best[0]
        ):
            best = item

    return (
        float(best[1]),
        float(best[2]),
    )


def best_f1_threshold(
    labels,
    probs,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    thresholds = np.unique(
        np.concatenate(
            (
                np.array([0.0, 1.0]),
                probs,
            )
        )
    )

    best = None

    for t in thresholds:
        m = confusion_metrics(
            labels,
            probs,
            float(t),
        )

        candidate = (
            m["f1"],
            m["accuracy"],
            -float(t),
            float(t),
        )

        if (
            best is None
            or candidate[:3] > best[:3]
        ):
            best = candidate

    return float(
        best[3]
    )


# ============================================================
# TRAIN / EVAL
# ============================================================

def make_train_loader(
    dataset,
    seed,
):
    labels = [
        int(row["label"])
        for row in dataset.rows
    ]

    counts = Counter(labels)

    weights = [
        1.0 / counts[label]
        for label in labels
    ]

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(
    dataset,
):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_optimizer(
    model,
    stage,
):
    if stage == "head":
        return torch.optim.AdamW(
            model.head.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )

    backbone_params = [
        p
        for p in model.wavlm.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR,
            },
            {
                "params": model.head.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
):
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    use_amp = (
        device.type == "cuda"
    )

    for batch_idx, batch in enumerate(
        loader,
        1,
    ):
        x = batch[
            "input_values"
        ].to(
            device,
            non_blocking=True,
        )

        mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                x,
                mask,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            scaled_loss = (
                loss
                / GRAD_ACCUM
            )

        scaler.scale(
            scaled_loss
        ).backward()

        if (
            batch_idx % GRAD_ACCUM == 0
            or batch_idx == len(loader)
        ):
            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += float(
            loss.detach().cpu()
        )

        n_batches += 1

    return (
        total_loss / max(
            n_batches,
            1,
        )
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    labels_all = []
    probs_all = []

    total_loss = 0.0
    n_batches = 0

    use_amp = (
        device.type == "cuda"
    )

    for batch in loader:
        x = batch[
            "input_values"
        ].to(
            device,
            non_blocking=True,
        )

        mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                x,
                mask,
            )

            loss = loss_fn(
                logits,
                labels,
            )

        probs = torch.sigmoid(
            logits.float()
        )

        labels_all.extend(
            labels.cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        probs_all.extend(
            probs.cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        total_loss += float(
            loss.detach().cpu()
        )

        n_batches += 1

    labels_np = np.asarray(
        labels_all,
        dtype=int,
    )

    probs_np = np.asarray(
        probs_all,
        dtype=float,
    )

    auc = roc_auc_score_np(
        labels_np,
        probs_np,
    )

    threshold = best_f1_threshold(
        labels_np,
        probs_np,
    )

    metrics = confusion_metrics(
        labels_np,
        probs_np,
        threshold,
    )

    eer, eer_threshold = eer_np(
        labels_np,
        probs_np,
    )

    metrics.update(
        {
            "loss": (
                total_loss
                / max(
                    n_batches,
                    1,
                )
            ),
            "roc_auc": auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "best_f1_threshold": threshold,
        }
    )

    return (
        metrics,
        labels_np,
        probs_np,
    )


def save_checkpoint(
    model,
    path,
    extra,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(
    model,
    path,
):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt[
        "model_state_dict"
    ]

    model.load_state_dict(
        state,
        strict=True,
    )

    return ckpt


def train_run(
    root: Path,
    rows,
    mode: str,
    seed: int,
    force=False,
):
    out_dir = (
        root
        / "experiments"
        / "wavlm_domain_robust"
        / mode
        / f"seed_{seed}"
    )

    metrics_path = (
        out_dir
        / "metrics.json"
    )

    best_path = (
        out_dir
        / "best_model.pt"
    )

    if (
        metrics_path.exists()
        and best_path.exists()
        and not force
    ):
        print(
            f"[SKIP] {mode} seed={seed}: ya terminado."
        )
        return

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(seed)

    train_rows = [
        r for r in rows
        if r.get("split") == "train"
    ]

    val_rows = [
        r for r in rows
        if r.get("split") == "val"
    ]

    test_rows = [
        r for r in rows
        if r.get("split") == "test"
    ]

    print()
    print("=" * 80)
    print(
        f"MODE={mode} | SEED={seed}"
    )
    print("=" * 80)
    print(
        f"train={len(train_rows)} "
        f"val={len(val_rows)} "
        f"test={len(test_rows)}"
    )

    train_ds = AudioDataset(
        train_rows,
        root,
        split="train",
        mode=mode,
    )

    val_ds = AudioDataset(
        val_rows,
        root,
        split="val",
        mode="control",
    )

    test_ds = AudioDataset(
        test_rows,
        root,
        split="test",
        mode="control",
    )

    train_loader = make_train_loader(
        train_ds,
        seed,
    )

    val_loader = make_eval_loader(
        val_ds
    )

    test_loader = make_eval_loader(
        test_ds
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"device={device}"
    )

    model = WavLMClassifier().to(
        device
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                device.type == "cuda"
            ),
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                device.type == "cuda"
            )
        )

    history = []

    best_auc = -float("inf")
    best_f1 = -float("inf")
    best_epoch = None
    best_stage = None
    epoch_global = 0

    # -----------------------
    # STAGE 1: HEAD ONLY
    # -----------------------
    freeze_backbone(model)

    trainable, total = count_trainable(
        model
    )

    print(
        f"HEAD stage: trainable "
        f"{trainable:,}/{total:,}"
    )

    optimizer = build_optimizer(
        model,
        stage="head",
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):
        epoch_global += 1

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch_global,
                "stage": "head",
                "train_loss": train_loss,
                **{
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                },
            }
        )

        print(
            f"[head {epoch}/{HEAD_EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"thr={val_metrics['best_f1_threshold']:.6f}"
        )

        auc = val_metrics[
            "roc_auc"
        ]

        f1 = val_metrics[
            "f1"
        ]

        better = (
            auc > best_auc + 1e-12
            or (
                abs(
                    auc - best_auc
                ) <= 1e-12
                and f1 > best_f1 + 1e-12
            )
        )

        if better:
            best_auc = auc
            best_f1 = f1
            best_epoch = epoch_global
            best_stage = "head"

            save_checkpoint(
                model,
                best_path,
                {
                    "mode": mode,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_val_auc": best_auc,
                    "best_val_f1": best_f1,
                },
            )

    # -----------------------
    # STAGE 2: LAST 4 LAYERS
    # -----------------------
    unfreeze_last_layers(
        model,
        LAST_N_LAYERS,
    )

    trainable, total = count_trainable(
        model
    )

    print(
        f"FINETUNE stage: trainable "
        f"{trainable:,}/{total:,}"
    )

    optimizer = build_optimizer(
        model,
        stage="finetune",
    )

    no_improve = 0

    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1,
    ):
        epoch_global += 1

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch_global,
                "stage": "finetune",
                "train_loss": train_loss,
                **{
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                },
            }
        )

        print(
            f"[ft {epoch}/{FINETUNE_EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"thr={val_metrics['best_f1_threshold']:.6f}"
        )

        auc = val_metrics[
            "roc_auc"
        ]

        f1 = val_metrics[
            "f1"
        ]

        better = (
            auc > best_auc + 1e-12
            or (
                abs(
                    auc - best_auc
                ) <= 1e-12
                and f1 > best_f1 + 1e-12
            )
        )

        if better:
            best_auc = auc
            best_f1 = f1
            best_epoch = epoch_global
            best_stage = "finetune"
            no_improve = 0

            save_checkpoint(
                model,
                best_path,
                {
                    "mode": mode,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_val_auc": best_auc,
                    "best_val_f1": best_f1,
                },
            )
        else:
            no_improve += 1

        if (
            no_improve
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping."
            )
            break

    # -----------------------
    # BEST CHECKPOINT
    # -----------------------
    ckpt = load_checkpoint(
        model,
        best_path,
    )

    val_metrics, val_labels, val_probs = evaluate(
        model,
        val_loader,
        device,
    )

    calibrated_threshold = best_f1_threshold(
        val_labels,
        val_probs,
    )

    test_metrics_05, test_labels, test_probs = evaluate(
        model,
        test_loader,
        device,
    )

    # evaluate() devuelve métricas en su threshold de mejor F1;
    # aquí calculamos explícitamente 0.5 y calibrated threshold.
    test_05 = confusion_metrics(
        test_labels,
        test_probs,
        0.5,
    )

    test_cal = confusion_metrics(
        test_labels,
        test_probs,
        calibrated_threshold,
    )

    test_auc = roc_auc_score_np(
        test_labels,
        test_probs,
    )

    test_eer, test_eer_thr = eer_np(
        test_labels,
        test_probs,
    )

    for d in (
        test_05,
        test_cal,
    ):
        d["roc_auc"] = test_auc
        d["eer"] = test_eer
        d[
            "eer_threshold"
        ] = test_eer_thr

    result = {
        "model": MODEL_NAME,
        "mode": mode,
        "seed": seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_val_auc": best_auc,
        "best_val_f1": best_f1,
        "calibrated_threshold": calibrated_threshold,
        "val_metrics_at_best_checkpoint": val_metrics,
        "test_threshold_0.5": test_05,
        "test_threshold_calibrated": test_cal,
        "augmentation": {
            "applied_to": "train_only",
            "label_independent": True,
            "gain_prob": GAIN_PROB,
            "gain_db": [
                GAIN_DB_MIN,
                GAIN_DB_MAX,
            ],
            "noise_prob": NOISE_PROB,
            "noise_snr_db": [
                NOISE_SNR_DB_MIN,
                NOISE_SNR_DB_MAX,
            ],
            "floor_prob": FLOOR_PROB,
            "floor_gate_probability_given_floor": FLOOR_GATE_PROB,
            "gate_dbfs": [
                GATE_DBFS_MIN,
                GATE_DBFS_MAX,
            ],
            "floor_noise_dbfs": [
                FLOOR_NOISE_DBFS_MIN,
                FLOOR_NOISE_DBFS_MAX,
            ],
            "lowpass_prob": LOWPASS_PROB,
            "lowpass_hz": [
                LOWPASS_HZ_MIN,
                LOWPASS_HZ_MAX,
            ],
            "reverb_prob": REVERB_PROB,
        }
        if mode == "robust"
        else {
            "applied_to": "none",
        },
        "history": history,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"BEST epoch={best_epoch} "
        f"stage={best_stage} "
        f"val_auc={best_auc:.4f} "
        f"thr={calibrated_threshold:.6f}"
    )

    print(
        "TEST calibrated: "
        f"acc={test_cal['accuracy']:.4f} "
        f"f1={test_cal['f1']:.4f} "
        f"recall={test_cal['recall']:.4f} "
        f"spec={test_cal['specificity']:.4f} "
        f"auc={test_auc:.4f}"
    )

    print(
        f"Guardado en:\n  {out_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Entrena WavLM control o robusto contra shortcuts "
            "de dominio, usando el mismo split interno."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "control",
            "robust",
        ],
        required=True,
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    metadata_path = (
        args.root
        / "data_normalized"
        / "metadata_normalized.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe:\n{metadata_path}"
        )

    rows = read_metadata(
        metadata_path
    )

    seeds = (
        SEEDS
        if args.all
        else [args.seed]
    )

    for seed in seeds:
        train_run(
            root=args.root,
            rows=rows,
            mode=args.mode,
            seed=seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
