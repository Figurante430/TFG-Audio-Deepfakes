from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import WavLMModel


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
# CHANNEL MIX V2
# ============================================================
# Se fija ANTES de volver a mirar Common Voice.
#
# 40 % de las muestras de train quedan intactas.
# 60 % reciben EXACTAMENTE UNA transformación de canal.
#
# Nada depende de la etiqueta: real y fake reciben la misma
# distribución de augmentations.
# ============================================================

AUGMENT_PROB = 0.60

# Probabilidades condicionales (suman 1.0)
P_NOISE_FLOOR = 0.40
P_REVERB = 0.25
P_LOWPASS = 0.20
P_HIGHPASS = 0.15

# Noise floor absoluto, útil para romper correlaciones de
# near-zero/silencios sin usar gate.
NOISE_FLOOR_DBFS_MIN = -55.0
NOISE_FLOOR_DBFS_MAX = -35.0

# Reverb sintética suave
REVERB_MAX_MS = 140.0
REVERB_TAPS_MIN = 2
REVERB_TAPS_MAX = 6
REVERB_DECAY_MIN = 0.15
REVERB_DECAY_MAX = 0.45

# Band-limit / canal
LOWPASS_HZ_MIN = 3500.0
LOWPASS_HZ_MAX = 7500.0

HIGHPASS_HZ_MIN = 60.0
HIGHPASS_HZ_MAX = 300.0

FIR_TAPS = 257


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


def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def read_metadata(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_audio_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def peak_protect(x, max_peak=0.995):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    peak = float(np.max(np.abs(x)))

    if peak > max_peak:
        x = x * (max_peak / peak)

    return x.astype(np.float32)


# ============================================================
# TRANSFORMACIONES DE CANAL
# ============================================================

def add_noise_floor(x):
    """
    Añade un ruido de fondo absoluto, no condicionado al RMS
    de la señal. Esto diversifica específicamente el noise floor.
    """
    x = np.asarray(x, dtype=np.float32)

    floor_dbfs = random.uniform(
        NOISE_FLOOR_DBFS_MIN,
        NOISE_FLOOR_DBFS_MAX,
    )

    sigma = db_to_amp(floor_dbfs)

    # Mezcla de ruido blanco y ruido suavizado para no crear
    # una única firma espectral.
    white = np.random.normal(
        0.0,
        sigma,
        size=len(x),
    ).astype(np.float32)

    if len(x) >= 5 and random.random() < 0.5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        colored = np.convolve(
            white,
            kernel,
            mode="same",
        ).astype(np.float32)

        noise = (
            0.5 * white
            + 0.5 * colored
        )
    else:
        noise = white

    return peak_protect(x + noise)


def synthetic_reverb(x):
    """
    Convolución con una RIR sintética dispersa y corta.
    No pretende modelar una sala concreta; solo randomizar canal.
    """
    x = np.asarray(x, dtype=np.float32)

    max_delay = max(
        1,
        int(
            TARGET_SR
            * REVERB_MAX_MS
            / 1000.0
        ),
    )

    n_taps = random.randint(
        REVERB_TAPS_MIN,
        REVERB_TAPS_MAX,
    )

    ir = np.zeros(
        max_delay + 1,
        dtype=np.float64,
    )
    ir[0] = 1.0

    decay = random.uniform(
        REVERB_DECAY_MIN,
        REVERB_DECAY_MAX,
    )

    delays = sorted(
        random.sample(
            range(1, max_delay + 1),
            k=min(
                n_taps,
                max_delay,
            ),
        )
    )

    for i, delay in enumerate(delays, 1):
        sign = -1.0 if random.random() < 0.25 else 1.0
        amp = (
            decay
            * (0.72 ** (i - 1))
            * random.uniform(0.6, 1.0)
        )
        ir[delay] = sign * amp

    y = np.convolve(
        x.astype(np.float64),
        ir,
        mode="full",
    )[:len(x)]

    return peak_protect(y)


def lowpass_fir(x, cutoff_hz):
    x = np.asarray(x, dtype=np.float32)

    taps = FIR_TAPS
    if taps % 2 == 0:
        taps += 1

    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / TARGET_SR

    h = 2.0 * fc * np.sinc(
        2.0 * fc * n
    )
    h *= np.hamming(taps)
    h /= np.sum(h)

    y = np.convolve(
        x.astype(np.float64),
        h.astype(np.float64),
        mode="same",
    )

    return peak_protect(y)


def highpass_fir(x, cutoff_hz):
    x = np.asarray(x, dtype=np.float32)

    taps = FIR_TAPS
    if taps % 2 == 0:
        taps += 1

    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / TARGET_SR

    lp = 2.0 * fc * np.sinc(
        2.0 * fc * n
    )
    lp *= np.hamming(taps)
    lp /= np.sum(lp)

    hp = -lp
    hp[(taps - 1) // 2] += 1.0

    y = np.convolve(
        x.astype(np.float64),
        hp.astype(np.float64),
        mode="same",
    )

    return peak_protect(y)


def channel_mix_augment(x):
    """
    40 % intacto.
    60 % exactamente UNA transformación.
    """
    x = np.asarray(x, dtype=np.float32)

    if random.random() >= AUGMENT_PROB:
        return x.copy()

    r = random.random()

    if r < P_NOISE_FLOOR:
        return add_noise_floor(x)

    r -= P_NOISE_FLOOR

    if r < P_REVERB:
        return synthetic_reverb(x)

    r -= P_REVERB

    if r < P_LOWPASS:
        cutoff = random.uniform(
            LOWPASS_HZ_MIN,
            LOWPASS_HZ_MAX,
        )
        return lowpass_fir(
            x,
            cutoff,
        )

    cutoff = random.uniform(
        HIGHPASS_HZ_MIN,
        HIGHPASS_HZ_MAX,
    )
    return highpass_fir(
        x,
        cutoff,
    )


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

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        if len(x) > MAX_SAMPLES:
            if self.split == "train":
                start = random.randint(
                    0,
                    len(x) - MAX_SAMPLES,
                )
            else:
                start = (
                    len(x) - MAX_SAMPLES
                ) // 2

            x = x[
                start:start + MAX_SAMPLES
            ]

        if (
            self.split == "train"
            and self.mode == "channel_mix"
        ):
            x = channel_mix_augment(x)

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

        return {
            "input_values": torch.from_numpy(
                x.astype(np.float32)
            ),
            "attention_mask": torch.from_numpy(mask),
            "label": torch.tensor(
                float(int(row["label"])),
                dtype=torch.float32,
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

        pooled = pooled / (
            feat_mask_f
            .sum(dim=1)
            .clamp_min(1.0)
        )

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

    for layer in model.wavlm.encoder.layers[
        -n_layers:
    ]:
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
# MÉTRICAS
# ============================================================

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

        avg = (
            (i + j - 1) / 2.0
            + 1.0
        )

        ranks[order[i:j]] = avg
        i = j

    return ranks


def roc_auc_np(labels, probs):
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

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(probs)

    return float(
        (
            ranks[pos].sum()
            - n_pos * (n_pos + 1) / 2.0
        )
        / (n_pos * n_neg)
    )


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

    accuracy = (
        (tp + tn) / len(labels)
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
        2.0
        * precision
        * recall
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


def best_f1_threshold(
    labels,
    probs,
):
    probs = np.asarray(
        probs,
        dtype=float,
    )

    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                probs,
            ]
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

    return float(best[3])


def eer_np(labels, probs):
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
            [
                np.array(
                    [-1e-9, 1.0 + 1e-9]
                ),
                probs,
            ]
        )
    )

    best = None

    for t in thresholds:
        m = confusion_metrics(
            labels,
            probs,
            float(t),
        )

        fpr = (
            m["fp"]
            / (m["fp"] + m["tn"])
            if m["fp"] + m["tn"]
            else 0.0
        )

        fnr = (
            m["fn"]
            / (m["fn"] + m["tp"])
            if m["fn"] + m["tp"]
            else 0.0
        )

        candidate = (
            abs(fpr - fnr),
            (fpr + fnr) / 2.0,
            float(t),
        )

        if (
            best is None
            or candidate[0] < best[0]
        ):
            best = candidate

    return (
        float(best[1]),
        float(best[2]),
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

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(dataset):
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
        ].to(device)

        mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

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
                loss / GRAD_ACCUM
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
        total_loss
        / max(
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
        ].to(device)

        mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

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

    threshold = best_f1_threshold(
        labels_np,
        probs_np,
    )

    metrics = confusion_metrics(
        labels_np,
        probs_np,
        threshold,
    )

    metrics["loss"] = (
        total_loss
        / max(
            n_batches,
            1,
        )
    )

    metrics["roc_auc"] = roc_auc_np(
        labels_np,
        probs_np,
    )

    eer, eer_thr = eer_np(
        labels_np,
        probs_np,
    )

    metrics["eer"] = eer
    metrics[
        "eer_threshold"
    ] = eer_thr
    metrics[
        "best_f1_threshold"
    ] = threshold

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
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return checkpoint


def train_run(
    root: Path,
    rows,
    seed: int,
    force=False,
):
    mode = "channel_mix"

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
            f"[SKIP] channel_mix seed={seed}: ya terminado."
        )
        return

    set_seed(seed)

    train_rows = [
        row
        for row in rows
        if row.get("split") == "train"
    ]

    val_rows = [
        row
        for row in rows
        if row.get("split") == "val"
    ]

    test_rows = [
        row
        for row in rows
        if row.get("split") == "test"
    ]

    print()
    print("=" * 80)
    print(
        f"CHANNEL-MIX V2 | SEED={seed}"
    )
    print("=" * 80)
    print(
        f"train={len(train_rows)} "
        f"val={len(val_rows)} "
        f"test={len(test_rows)}"
    )
    print(
        f"augment_prob={AUGMENT_PROB} | "
        "exactamente una transformación cuando se aplica"
    )

    train_ds = AudioDataset(
        train_rows,
        root,
        split="train",
        mode="channel_mix",
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

    # Stage 1
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
        "head",
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
                    for k, v
                    in val_metrics.items()
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

    # Stage 2
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
        "finetune",
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
                    for k, v
                    in val_metrics.items()
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

    load_checkpoint(
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

    _, test_labels, test_probs = evaluate(
        model,
        test_loader,
        device,
    )

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

    test_auc = roc_auc_np(
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
        "test_threshold_0.5": test_05,
        "test_threshold_calibrated": test_cal,
        "augmentation": {
            "applied_to": "train_only",
            "label_independent": True,
            "scheme": "channel_mix_v2_exactly_one",
            "augment_probability": AUGMENT_PROB,
            "conditional_probabilities": {
                "noise_floor": P_NOISE_FLOOR,
                "reverb": P_REVERB,
                "lowpass": P_LOWPASS,
                "highpass": P_HIGHPASS,
            },
            "noise_floor_dbfs": [
                NOISE_FLOOR_DBFS_MIN,
                NOISE_FLOOR_DBFS_MAX,
            ],
            "lowpass_hz": [
                LOWPASS_HZ_MIN,
                LOWPASS_HZ_MAX,
            ],
            "highpass_hz": [
                HIGHPASS_HZ_MIN,
                HIGHPASS_HZ_MAX,
            ],
            "reverb_max_ms": REVERB_MAX_MS,
        },
        "history": history,
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "WavLM con domain randomization de canal "
            "controlada y label-independent."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
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
            args.root,
            rows,
            seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
