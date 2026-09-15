from __future__ import annotations
import argparse, csv, gc, json, random, statistics
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
METADATA_CSV = ROOT / "data_normalized" / "metadata_normalized.csv"
OUTPUT_ROOT = ROOT / "experiments" / "loso_wavlm_3seeds"
MODEL_NAME = "microsoft/wavlm-base-plus"

SEEDS = [42, 123, 2026]
SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
NUM_WORKERS = 0

HEAD_EPOCHS = 2
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 8
FINETUNE_LAST_N_LAYERS = 4
BACKBONE_LR = 1e-5
HEAD_FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3
USE_AMP = True


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(METADATA_CSV)

    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"path", "label", "split", "generator", "speaker_id", "utterance_id"}
    if not rows:
        raise RuntimeError("metadata_normalized.csv está vacío.")

    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError("Faltan columnas: " + ", ".join(sorted(missing)))

    return [r for r in rows if r["split"] in {"train", "val", "test"}]


def get_speakers(rows):
    speakers = sorted({
        r["speaker_id"].strip().lower()
        for r in rows
        if r["speaker_id"].strip()
        and r["speaker_id"].strip().lower() != "unknown"
    })
    if not speakers:
        raise RuntimeError("No hay speaker_id válidos.")
    return speakers


def resolve_path(text):
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


class AudioDataset(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        audio, sr = sf.read(
            str(resolve_path(row["path"])),
            dtype="float32",
            always_2d=True,
        )

        if sr != SAMPLE_RATE:
            raise RuntimeError(
                f"{row['path']}: sample rate={sr}; esperaba {SAMPLE_RATE}"
            )

        waveform = torch.from_numpy(audio.mean(axis=1)).float()
        n = waveform.numel()

        if n >= NUM_SAMPLES:
            max_start = n - NUM_SAMPLES
            if self.training and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2
            waveform = waveform[start:start + NUM_SAMPLES]
            mask = torch.ones(NUM_SAMPLES, dtype=torch.long)
        else:
            missing = NUM_SAMPLES - n
            left = missing // 2
            right = missing - left
            waveform = F.pad(waveform, (left, right))
            mask = torch.zeros(NUM_SAMPLES, dtype=torch.long)
            mask[left:left+n] = 1

        return {
            "input_values": waveform,
            "attention_mask": mask,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "path": row["path"],
            "generator": row["generator"],
            "speaker_id": row["speaker_id"],
        }


class WavLMDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )

    def forward(self, input_values, attention_mask):
        out = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state

        if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
            fmask = self.backbone._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            m = fmask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)

        return self.classifier(pooled).squeeze(1)


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_layers(model, n):
    for p in model.backbone.parameters():
        p.requires_grad = False

    layers = model.backbone.encoder.layers
    for layer in layers[-min(n, len(layers)):]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(model.backbone.encoder, "layer_norm"):
        for p in model.backbone.encoder.layer_norm.parameters():
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def binary_metrics(labels, probs, threshold=0.5):
    labels = [int(x) for x in labels]
    preds = [1 if p >= threshold else 0 for p in probs]

    tp = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, preds))

    total = len(labels)
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_points(labels, probs):
    pairs = sorted(zip(probs, labels), reverse=True)
    pos = sum(int(y) == 1 for y in labels)
    neg = sum(int(y) == 0 for y in labels)

    if pos == 0 or neg == 0:
        return []

    tp = fp = 0
    points = [(0.0, 0.0, float("inf"))]
    last = None

    for score, label in pairs:
        if last is not None and score != last:
            points.append((fp / neg, tp / pos, last))
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        last = score

    points.append((fp / neg, tp / pos, last if last is not None else 0.0))
    return points


def roc_auc(labels, probs):
    pts = sorted(roc_points(labels, probs), key=lambda x: x[0])
    if len(pts) < 2:
        return float("nan")

    auc = 0.0
    for i in range(1, len(pts)):
        x1, y1, _ = pts[i-1]
        x2, y2, _ = pts[i]
        auc += (x2 - x1) * (y1 + y2) / 2.0
    return auc


def eer(labels, probs):
    pts = roc_points(labels, probs)
    if not pts:
        return float("nan"), float("nan")

    best = None
    for fpr, tpr, threshold in pts:
        fnr = 1.0 - tpr
        item = (abs(fpr - fnr), (fpr + fnr) / 2.0, threshold)
        if best is None or item[0] < best[0]:
            best = item
    return best[1], best[2]


def best_f1_threshold(labels, probs):
    candidates = [0.0] + sorted(set(float(x) for x in probs)) + [1.0]
    best_t, best_f1 = 0.5, -1.0

    for t in candidates:
        f1 = binary_metrics(labels, probs, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    return best_t, best_f1


def make_loader(rows, training):
    ds = AudioDataset(rows, training=training)

    if not training:
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    labels = [int(r["label"]) for r in rows]
    counts = Counter(labels)

    if 0 not in counts or 1 not in counts:
        raise RuntimeError(f"Train necesita ambas clases: {dict(counts)}")

    class_weight = {k: 1.0 / v for k, v in counts.items()}
    weights = torch.tensor(
        [class_weight[y] for y in labels],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(ds),
        replacement=True,
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def make_scaler(device):
    enabled = USE_AMP and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=USE_AMP and device.type == "cuda",
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    labels_all, probs_all = [], []
    total_loss = total = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, 1):
        x = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device):
            logits = model(x, mask)
            raw_loss = criterion(logits, labels)
            loss = raw_loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if batch_idx % GRAD_ACCUM_STEPS == 0 or batch_idx == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(logits).detach().cpu().tolist()
        labels_cpu = labels.detach().cpu().tolist()

        labels_all.extend(labels_cpu)
        probs_all.extend(probs)

        bs = labels.size(0)
        total_loss += raw_loss.item() * bs
        total += bs

    m = binary_metrics(labels_all, probs_all)
    m["loss"] = total_loss / total
    m["roc_auc"] = roc_auc(labels_all, probs_all)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, predictions=False):
    model.eval()

    labels_all, probs_all, rows = [], [], []
    total_loss = total = 0

    for batch in loader:
        x = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device):
            logits = model(x, mask)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).detach().cpu().tolist()
        labels_cpu = labels.detach().cpu().tolist()

        labels_all.extend(labels_cpu)
        probs_all.extend(probs)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs

        if predictions:
            for i in range(bs):
                rows.append({
                    "path": batch["path"][i],
                    "label": int(labels_cpu[i]),
                    "prob_fake": float(probs[i]),
                    "prediction": 1 if probs[i] >= threshold else 0,
                    "generator": batch["generator"][i],
                    "speaker_id": batch["speaker_id"][i],
                })

    m = binary_metrics(labels_all, probs_all, threshold)
    m["loss"] = total_loss / total if total else 0.0
    m["roc_auc"] = roc_auc(labels_all, probs_all)
    m["eer"], m["eer_threshold"] = eer(labels_all, probs_all)

    return m, labels_all, probs_all, rows


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(all_rows, heldout, seed, device, force=False):
    seed_everything(seed)

    run_dir = OUTPUT_ROOT / f"holdout_{heldout}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "test_predictions.csv"

    if metrics_path.exists() and not force:
        print(f"[SKIP] {heldout} seed={seed}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))["summary_row"]

    print("\n" + "=" * 86)
    print(f"WavLM LOSO | SPEAKER NO VISTO = {heldout} | seed={seed}")
    print("=" * 86)

    # Mantenemos los splits originales para conservar el aislamiento
    # de utterance_id entre train/val/test.
    train_rows = [
        r for r in all_rows
        if r["split"] == "train"
        and r["speaker_id"].strip().lower() != heldout
    ]
    val_rows = [
        r for r in all_rows
        if r["split"] == "val"
        and r["speaker_id"].strip().lower() != heldout
    ]
    test_rows = [
        r for r in all_rows
        if r["split"] == "test"
        and r["speaker_id"].strip().lower() == heldout
    ]

    if any(r["speaker_id"].strip().lower() == heldout for r in train_rows + val_rows):
        raise RuntimeError(f"Leakage de speaker: {heldout}")

    test_counts = Counter(int(r["label"]) for r in test_rows)

    if not test_rows or 0 not in test_counts or 1 not in test_counts:
        raise RuntimeError(
            f"Test de {heldout} necesita reales y fakes. Counts={dict(test_counts)}"
        )

    print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
    print(f"Test real={test_counts[0]} fake={test_counts[1]}")
    print("Test generators:", dict(sorted(Counter(r["generator"] for r in test_rows).items())))

    train_loader = make_loader(train_rows, True)
    val_loader = make_loader(val_rows, False)
    test_loader = make_loader(test_rows, False)

    model = WavLMDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_val_auc = -1.0
    best_val_f1 = -1.0
    global_epoch = 0

    freeze_backbone(model)
    tr, total = count_trainable(model)
    print(f"Fase HEAD | trainable={tr:,}/{total:,}")

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = make_scaler(device)

    for _ in range(HEAD_EPOCHS):
        global_epoch += 1
        tm = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vm, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"[{heldout} seed={seed}] HEAD {global_epoch:02d} | "
            f"train_f1={tm['f1']:.4f} val_f1={vm['f1']:.4f} "
            f"val_auc={vm['roc_auc']:.4f} val_eer={vm['eer']:.4f}"
        )

        history.append({
            "seed": seed, "epoch": global_epoch, "stage": "head",
            "train_loss": tm["loss"], "train_f1": tm["f1"], "train_auc": tm["roc_auc"],
            "val_loss": vm["loss"], "val_f1": vm["f1"],
            "val_auc": vm["roc_auc"], "val_eer": vm["eer"],
        })
        write_csv(history_path, history)

        better = (
            vm["roc_auc"] > best_val_auc
            or (vm["roc_auc"] == best_val_auc and vm["f1"] > best_val_f1)
        )

        if better:
            best_val_auc = vm["roc_auc"]
            best_val_f1 = vm["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": global_epoch,
                "stage": "head",
                "held_out_speaker": heldout,
                "seed": seed,
                "val_auc": best_val_auc,
                "val_f1": best_val_f1,
            }, best_model_path)

    unfreeze_last_layers(model, FINETUNE_LAST_N_LAYERS)
    tr, total = count_trainable(model)
    print(f"Fase FINETUNE({FINETUNE_LAST_N_LAYERS}) | trainable={tr:,}/{total:,}")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_FINETUNE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scaler = make_scaler(device)

    no_improve = 0

    for _ in range(FINETUNE_EPOCHS):
        global_epoch += 1
        tm = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vm, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"[{heldout} seed={seed}] FT {global_epoch:02d} | "
            f"train_f1={tm['f1']:.4f} val_f1={vm['f1']:.4f} "
            f"val_auc={vm['roc_auc']:.4f} val_eer={vm['eer']:.4f}"
        )

        history.append({
            "seed": seed, "epoch": global_epoch, "stage": "finetune",
            "train_loss": tm["loss"], "train_f1": tm["f1"], "train_auc": tm["roc_auc"],
            "val_loss": vm["loss"], "val_f1": vm["f1"],
            "val_auc": vm["roc_auc"], "val_eer": vm["eer"],
        })
        write_csv(history_path, history)

        better = (
            vm["roc_auc"] > best_val_auc
            or (vm["roc_auc"] == best_val_auc and vm["f1"] > best_val_f1)
        )

        if better:
            best_val_auc = vm["roc_auc"]
            best_val_f1 = vm["f1"]
            no_improve = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": global_epoch,
                "stage": "finetune",
                "held_out_speaker": heldout,
                "seed": seed,
                "val_auc": best_val_auc,
                "val_f1": best_val_f1,
            }, best_model_path)
            print("  BEST")
        else:
            no_improve += 1
            print(f"  Sin mejora: {no_improve}/{EARLY_STOPPING_PATIENCE}")
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("  Early stopping.")
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_labels, val_probs, _ = evaluate(
        model, val_loader, criterion, device
    )
    threshold, val_cal_f1 = best_f1_threshold(val_labels, val_probs)

    metrics_05, _, _, _ = evaluate(
        model, test_loader, criterion, device, threshold=0.5
    )
    metrics_cal, _, _, pred_rows = evaluate(
        model, test_loader, criterion, device,
        threshold=threshold,
        predictions=True,
    )

    for row in pred_rows:
        row["prediction_calibrated"] = row.pop("prediction")
        row["prediction_05"] = 1 if row["prob_fake"] >= 0.5 else 0
        row["threshold_calibrated"] = threshold
        row["seed"] = seed
        row["held_out_speaker"] = heldout

    write_csv(predictions_path, pred_rows)

    summary_row = {
        "held_out_speaker": heldout,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_stage": checkpoint["stage"],
        "best_val_auc": float(checkpoint["val_auc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
        "threshold_val": float(threshold),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "n_test_real": test_counts[0],
        "n_test_fake": test_counts[1],

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

    metrics_path.write_text(
        json.dumps({
            "model": MODEL_NAME,
            "experiment": "LOSO",
            "held_out_speaker": heldout,
            "seed": seed,
            "calibrated_threshold": threshold,
            "val_best_f1_at_calibrated_threshold": val_cal_f1,
            "threshold_0.5": metrics_05,
            "threshold_calibrated": metrics_cal,
            "summary_row": summary_row,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nTEST LOSO [{heldout}] seed={seed}\n"
        f"  threshold=0.5      | f1={metrics_05['f1']:.4f} "
        f"auc={metrics_05['roc_auc']:.4f} rec={metrics_05['recall']:.4f} "
        f"spec={metrics_05['specificity']:.4f} eer={metrics_05['eer']:.4f}\n"
        f"  threshold={threshold:.6f} | f1={metrics_cal['f1']:.4f} "
        f"auc={metrics_cal['roc_auc']:.4f} rec={metrics_cal['recall']:.4f} "
        f"spec={metrics_cal['specificity']:.4f} eer={metrics_cal['eer']:.4f}"
    )

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_row


def mean_std(values):
    vals = [float(v) for v in values]
    return (
        statistics.mean(vals),
        statistics.stdev(vals) if len(vals) > 1 else 0.0,
    )


def aggregate(rows, speakers):
    result = []

    for speaker in speakers:
        group = [r for r in rows if r["held_out_speaker"] == speaker]
        if not group:
            continue

        out = {
            "held_out_speaker": speaker,
            "n_seeds": len(group),
            "seeds": ",".join(str(r["seed"]) for r in group),
            "n_test": group[0].get("n_test", ""),
            "n_test_real": group[0].get("n_test_real", ""),
            "n_test_fake": group[0].get("n_test_fake", ""),
        }

        for metric in [
            "acc_05", "precision_05", "recall_05", "specificity_05",
            "f1_05", "auc_05", "eer_05",
            "acc_cal", "precision_cal", "recall_cal", "specificity_cal",
            "f1_cal", "auc_cal", "eer_cal", "threshold_val",
        ]:
            mean, std = mean_std([r[metric] for r in group])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std

        result.append(out)

    return result


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--speaker")
    g.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_metadata()
    speakers = get_speakers(rows)

    print("=" * 86)
    print("WavLM LOSO - 3 SEMILLAS")
    print("=" * 86)
    print("Speakers:", speakers)
    print("Seeds:", SEEDS)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if args.all:
        folds = speakers
    elif args.speaker:
        requested = args.speaker.strip().lower()
        if requested not in speakers:
            raise RuntimeError(
                f"Speaker '{requested}' no existe. Disponibles: {speakers}"
            )
        folds = [requested]
    else:
        folds = ["gonzalo"] if "gonzalo" in speakers else [speakers[0]]

    all_runs_path = OUTPUT_ROOT / "summary_all_runs.csv"

    previous = []
    if all_runs_path.exists():
        with all_runs_path.open("r", encoding="utf-8-sig", newline="") as f:
            previous = list(csv.DictReader(f))

    by_key = {
        (r["held_out_speaker"], str(r["seed"])): r
        for r in previous
    }

    for speaker in folds:
        for seed in SEEDS:
            row = run_one(
                rows,
                speaker,
                seed,
                device,
                force=args.force,
            )
            by_key[(row["held_out_speaker"], str(row["seed"]))] = row

            merged = list(by_key.values())
            write_csv(all_runs_path, merged)
            write_csv(
                OUTPUT_ROOT / "summary_mean_std.csv",
                aggregate(merged, speakers),
            )

    merged = list(by_key.values())
    agg = aggregate(merged, speakers)
    write_csv(OUTPUT_ROOT / "summary_mean_std.csv", agg)

    print("\n" + "=" * 86)
    print("RESUMEN LOSO MEDIA ± STD (threshold calibrado)")
    print("=" * 86)

    for r in agg:
        print(
            f"{r['held_out_speaker']:12s} | "
            f"F1={float(r['f1_cal_mean']):.4f} ± {float(r['f1_cal_std']):.4f} | "
            f"AUC={float(r['auc_cal_mean']):.4f} ± {float(r['auc_cal_std']):.4f} | "
            f"EER={float(r['eer_cal_mean']):.4f} ± {float(r['eer_cal_std']):.4f} | "
            f"Recall={float(r['recall_cal_mean']):.4f} ± {float(r['recall_cal_std']):.4f} | "
            f"Spec={float(r['specificity_cal_mean']):.4f} ± {float(r['specificity_cal_std']):.4f}"
        )

    print("\nRuns:", all_runs_path)
    print("Media/std:", OUTPUT_ROOT / "summary_mean_std.csv")


if __name__ == "__main__":
    main()
