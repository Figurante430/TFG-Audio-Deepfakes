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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


class AudioDataset(Dataset):
    def __init__(self, rows, root, split):
        self.rows = list(rows)
        self.root = Path(root)
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = resolve_path(self.root, row["path"])

        x, sr = sf.read(path, dtype="float32", always_2d=False)

        if x.ndim == 2:
            x = x.mean(axis=1)

        if sr != TARGET_SR:
            raise RuntimeError(f"{path}: {sr} Hz, esperaba {TARGET_SR} Hz")

        x = np.asarray(x, dtype=np.float32)

        if len(x) > MAX_SAMPLES:
            if self.split == "train":
                start = random.randint(0, len(x) - MAX_SAMPLES)
            else:
                start = (len(x) - MAX_SAMPLES) // 2
            x = x[start:start + MAX_SAMPLES]

        valid_len = min(len(x), MAX_SAMPLES)

        if len(x) < MAX_SAMPLES:
            x = np.pad(x, (0, MAX_SAMPLES - len(x)), mode="constant")

        mask = np.zeros(MAX_SAMPLES, dtype=np.int64)
        mask[:valid_len] = 1

        return {
            "input_values": torch.from_numpy(x.astype(np.float32)),
            "attention_mask": torch.from_numpy(mask),
            "label": torch.tensor(float(int(row["label"])), dtype=torch.float32),
        }


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

        m = feat_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        return self.head(pooled).squeeze(-1)


def freeze_backbone(model):
    for p in model.wavlm.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_last_layers(model):
    for p in model.wavlm.parameters():
        p.requires_grad = False

    for layer in model.wavlm.encoder.layers[-LAST_N_LAYERS:]:
        for p in layer.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j

    return ranks


def roc_auc_np(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if not n_pos or not n_neg:
        return float("nan")

    ranks = rankdata(probs)
    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def confusion_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))

    acc = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": spec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(labels, probs):
    probs = np.asarray(probs, dtype=float)
    thresholds = np.unique(np.concatenate([np.array([0.0, 1.0]), probs]))

    best = None

    for t in thresholds:
        m = confusion_metrics(labels, probs, float(t))
        candidate = (m["f1"], m["accuracy"], -float(t), float(t))
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    return float(best[3])


def eer_np(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    thresholds = np.unique(
        np.concatenate([np.array([-1e-9, 1.0 + 1e-9]), probs])
    )

    best = None

    for t in thresholds:
        m = confusion_metrics(labels, probs, float(t))
        fpr = m["fp"] / (m["fp"] + m["tn"]) if m["fp"] + m["tn"] else 0.0
        fnr = m["fn"] / (m["fn"] + m["tp"]) if m["fn"] + m["tp"] else 0.0
        item = (abs(fpr - fnr), (fpr + fnr) / 2.0, float(t))

        if best is None or item[0] < best[0]:
            best = item

    return float(best[1]), float(best[2])


def make_domain_balanced_loader(dataset, seed):
    """
    Muestreo esperado por batch/época:
      50 % fake interno
      25 % real interno
      25 % real externo Common Voice

    Así Common Voice no domina simplemente por cantidad.
    """
    groups = []

    for row in dataset.rows:
        label = int(row["label"])
        domain = row.get("_domain", "internal")

        if label == 1:
            groups.append("fake_internal")
        elif domain == "external_adapt":
            groups.append("real_external")
        else:
            groups.append("real_internal")

    counts = Counter(groups)

    required = {"fake_internal", "real_internal", "real_external"}
    missing = required - set(counts)

    if missing:
        raise RuntimeError(f"Faltan grupos para el sampler: {sorted(missing)}")

    group_mass = {
        "fake_internal": 0.50,
        "real_internal": 0.25,
        "real_external": 0.25,
    }

    weights = [
        group_mass[g] / counts[g]
        for g in groups
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset.rows),
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


def build_optimizer(model, stage):
    if stage == "head":
        return torch.optim.AdamW(
            model.head.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )

    backbone = [p for p in model.wavlm.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)

    total = 0.0
    n = 0
    use_amp = device.type == "cuda"

    for i, batch in enumerate(loader, 1):
        x = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            scaled_loss = loss / GRAD_ACCUM

        scaler.scale(scaled_loss).backward()

        if i % GRAD_ACCUM == 0 or i == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.detach().cpu())
        n += 1

    return total / max(n, 1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    labels = []
    probs = []
    total = 0.0
    n = 0
    use_amp = device.type == "cuda"

    for batch in loader:
        x = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(x, mask)
            loss = loss_fn(logits, y)

        p = torch.sigmoid(logits.float())

        labels.extend(y.cpu().numpy().astype(int).tolist())
        probs.extend(p.cpu().numpy().astype(float).tolist())

        total += float(loss.detach().cpu())
        n += 1

    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    thr = best_f1_threshold(labels, probs)
    metrics = confusion_metrics(labels, probs, thr)
    metrics["loss"] = total / max(n, 1)
    metrics["roc_auc"] = roc_auc_np(labels, probs)

    eer, eer_thr = eer_np(labels, probs)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_thr
    metrics["best_f1_threshold"] = thr

    return metrics, labels, probs


def save_checkpoint(model, path, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), **extra},
        path,
    )


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


def train_seed(root, internal_rows, external_rows, seed, force=False):
    out_dir = (
        root
        / "experiments"
        / "wavlm_domain_adapt_real"
        / f"seed_{seed}"
    )
    best_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "metrics.json"

    if best_path.exists() and metrics_path.exists() and not force:
        print(f"[SKIP] seed={seed}: ya terminado")
        return

    set_seed(seed)

    internal_train = [
        dict(r, _domain="internal")
        for r in internal_rows
        if r.get("split") == "train"
    ]

    val_rows = [
        dict(r, _domain="internal")
        for r in internal_rows
        if r.get("split") == "val"
    ]

    test_rows = [
        dict(r, _domain="internal")
        for r in internal_rows
        if r.get("split") == "test"
    ]

    external_train = [
        dict(r, label="0", split="train", _domain="external_adapt")
        for r in external_rows
    ]

    train_rows = internal_train + external_train

    print()
    print("=" * 80)
    print(f"DOMAIN ADAPT REAL | seed={seed}")
    print("=" * 80)
    print(f"Internal train : {len(internal_train)}")
    print(f"External reals : {len(external_train)}")
    print(f"Validation     : {len(val_rows)} (solo interna)")
    print(f"Test interno   : {len(test_rows)}")

    train_ds = AudioDataset(train_rows, root, "train")
    val_ds = AudioDataset(val_rows, root, "val")
    test_ds = AudioDataset(test_rows, root, "test")

    train_loader = make_domain_balanced_loader(train_ds, seed)
    val_loader = make_eval_loader(val_ds)
    test_loader = make_eval_loader(test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = WavLMClassifier().to(device)

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda"),
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda")
        )

    best_auc = -float("inf")
    best_f1 = -float("inf")
    best_epoch = None
    best_stage = None
    history = []
    epoch_global = 0

    freeze_backbone(model)
    optimizer = build_optimizer(model, "head")

    for epoch in range(1, HEAD_EPOCHS + 1):
        epoch_global += 1

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch_global,
            "stage": "head",
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"[head {epoch}/{HEAD_EPOCHS}] "
            f"loss={train_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"thr={val_metrics['best_f1_threshold']:.6f}"
        )

        auc = val_metrics["roc_auc"]
        f1 = val_metrics["f1"]

        better = (
            auc > best_auc + 1e-12
            or (abs(auc - best_auc) <= 1e-12 and f1 > best_f1 + 1e-12)
        )

        if better:
            best_auc, best_f1 = auc, f1
            best_epoch, best_stage = epoch_global, "head"
            save_checkpoint(
                model,
                best_path,
                {
                    "seed": seed,
                    "mode": "domain_adapt_real",
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_val_auc": best_auc,
                    "best_val_f1": best_f1,
                },
            )

    unfreeze_last_layers(model)
    optimizer = build_optimizer(model, "finetune")

    no_improve = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        epoch_global += 1

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch_global,
            "stage": "finetune",
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"[ft {epoch}/{FINETUNE_EPOCHS}] "
            f"loss={train_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"thr={val_metrics['best_f1_threshold']:.6f}"
        )

        auc = val_metrics["roc_auc"]
        f1 = val_metrics["f1"]

        better = (
            auc > best_auc + 1e-12
            or (abs(auc - best_auc) <= 1e-12 and f1 > best_f1 + 1e-12)
        )

        if better:
            best_auc, best_f1 = auc, f1
            best_epoch, best_stage = epoch_global, "finetune"
            no_improve = 0

            save_checkpoint(
                model,
                best_path,
                {
                    "seed": seed,
                    "mode": "domain_adapt_real",
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_val_auc": best_auc,
                    "best_val_f1": best_f1,
                },
            )
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    load_checkpoint(model, best_path)

    _, val_labels, val_probs = evaluate(model, val_loader, device)
    calibrated_threshold = best_f1_threshold(val_labels, val_probs)

    _, test_labels, test_probs = evaluate(model, test_loader, device)

    test_cal = confusion_metrics(
        test_labels,
        test_probs,
        calibrated_threshold,
    )
    test_auc = roc_auc_np(test_labels, test_probs)
    test_eer, test_eer_thr = eer_np(test_labels, test_probs)

    test_cal["roc_auc"] = test_auc
    test_cal["eer"] = test_eer
    test_cal["eer_threshold"] = test_eer_thr

    result = {
        "model": MODEL_NAME,
        "mode": "domain_adapt_real",
        "seed": seed,
        "n_internal_train": len(internal_train),
        "n_external_real_train": len(external_train),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "sampler_expected_mass": {
            "fake_internal": 0.50,
            "real_internal": 0.25,
            "real_external": 0.25,
        },
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_val_auc": best_auc,
        "best_val_f1": best_f1,
        "calibrated_threshold": calibrated_threshold,
        "test_threshold_calibrated": test_cal,
        "history": history,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print(
        f"BEST epoch={best_epoch} stage={best_stage} "
        f"val_auc={best_auc:.4f} thr={calibrated_threshold:.6f}"
    )
    print(
        "TEST calibrated: "
        f"acc={test_cal['accuracy']:.4f} "
        f"f1={test_cal['f1']:.4f} "
        f"recall={test_cal['recall']:.4f} "
        f"spec={test_cal['specificity']:.4f} "
        f"auc={test_auc:.4f}"
    )
    print(f"Guardado en:\n  {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--external-metadata",
        type=Path,
        default=ROOT_DEFAULT / "external_adaptation" / "metadata_external_adapt_train.csv",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--all", action="store_true")

    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    internal_meta = args.root / "data_normalized" / "metadata_normalized.csv"

    if not internal_meta.exists():
        raise FileNotFoundError(internal_meta)

    if not args.external_metadata.exists():
        raise FileNotFoundError(
            f"No existe el metadata externo de adaptación:\n{args.external_metadata}"
        )

    internal_rows = read_csv(internal_meta)
    external_rows = read_csv(args.external_metadata)

    seeds = SEEDS if args.all else [args.seed]

    for seed in seeds:
        train_seed(
            args.root,
            internal_rows,
            external_rows,
            seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
