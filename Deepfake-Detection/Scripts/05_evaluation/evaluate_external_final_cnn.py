from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection"
)

METADATA_CSV = (
    ROOT
    / "external_final_holdout"
    / "metadata_external_final.csv"
)

FREEZE_MANIFEST = (
    ROOT
    / "external_final_holdout"
    / "freeze_manifest_external_final.json"
)

EXPECTED_FINGERPRINT = (
    "3fed1b260a404af75b21aab8a57f1cf"
    "2dd0dbc8967d2695b45780095008f6021"
)

CHECKPOINT = (
    ROOT
    / "experiments"
    / "baseline_logmel_cnn"
    / "best_model.pt"
)

OUTPUT_DIR = (
    ROOT
    / "external_results"
    / "baseline_cnn_final"
)

PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "cnn_predictions.csv"
)

METRICS_JSON = (
    OUTPUT_DIR
    / "cnn_metrics.json"
)

BY_GENERATOR_CSV = (
    OUTPUT_DIR
    / "cnn_by_generator.csv"
)


# =====================================================================
# MISMA CONFIGURACIÓN DEL ENTRENAMIENTO
# =====================================================================

SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
NUM_SAMPLES = int(
    SAMPLE_RATE * CLIP_SECONDS
)

N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256

FIXED_THRESHOLD = 0.5

EXPECTED_TOTAL = 220
EXPECTED_REAL = 100
EXPECTED_FAKE = 120


# =====================================================================
# UTILS
# =====================================================================

def fail(message):

    print()
    print("=" * 76)
    print("ERROR")
    print("=" * 76)
    print(message)
    print("=" * 76)

    sys.exit(1)


def sha256(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def read_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# VERIFY HOLDOUT
# =====================================================================

def verify_holdout():

    if not METADATA_CSV.exists():
        fail(
            f"No existe:\n{METADATA_CSV}"
        )

    if not FREEZE_MANIFEST.exists():
        fail(
            f"No existe:\n{FREEZE_MANIFEST}"
        )


    with FREEZE_MANIFEST.open(
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)


    fingerprint = manifest.get(
        "dataset_fingerprint_sha256"
    )


    if fingerprint != EXPECTED_FINGERPRINT:

        fail(
            "Fingerprint diferente.\n\n"
            f"Esperado:\n{EXPECTED_FINGERPRINT}\n\n"
            f"Actual:\n{fingerprint}"
        )


    rows = read_csv(
        METADATA_CSV
    )


    if len(rows) != EXPECTED_TOTAL:

        fail(
            f"Hay {len(rows)} audios; "
            f"esperaba {EXPECTED_TOTAL}."
        )


    real = sum(
        int(row["label"]) == 0
        for row in rows
    )

    fake = sum(
        int(row["label"]) == 1
        for row in rows
    )


    if real != EXPECTED_REAL:
        fail(
            f"REAL={real}; "
            f"esperaba {EXPECTED_REAL}."
        )

    if fake != EXPECTED_FAKE:
        fail(
            f"FAKE={fake}; "
            f"esperaba {EXPECTED_FAKE}."
        )


    print()
    print("=" * 76)
    print("VERIFICANDO HOLDOUT")
    print("=" * 76)


    for i, row in enumerate(
        rows,
        start=1
    ):

        path = (
            ROOT
            / Path(
                row["relative_path"]
            )
        )


        if not path.exists():

            fail(
                f"Falta:\n{path}"
            )


        current = sha256(
            path
        ).lower()

        expected = (
            row["sha256"]
            .strip()
            .lower()
        )


        if current != expected:

            fail(
                "WAV MODIFICADO:\n"
                f"{path}\n\n"
                f"Esperado: {expected}\n"
                f"Actual  : {current}"
            )


        if (
            i % 25 == 0
            or i == len(rows)
        ):

            print(
                f"\rVerificados: {i}/{len(rows)}",
                end="",
                flush=True
            )


    print()
    print(
        f"Fingerprint: {fingerprint}"
    )

    print("Integridad : OK")


    return rows


# =====================================================================
# AUDIO
#
# Misma lógica que AudioDataset del entrenamiento:
#
# - soundfile
# - convertir a mono
# - resample torchaudio a 16 kHz
# - crop central en evaluación
# - padding simétrico
# =====================================================================

def load_audio(path: Path):

    audio, sr = sf.read(

        str(path),

        dtype="float32",

        always_2d=True
    )


    if audio.size == 0:

        raise RuntimeError(
            f"Audio vacío: {path}"
        )


    audio = (
        audio.mean(
            axis=1
        )
    )


    waveform = (
        torch.from_numpy(
            audio
        )
        .float()
    )


    if sr != SAMPLE_RATE:

        waveform = (
            torchaudio.functional.resample(

                waveform.unsqueeze(0),

                sr,

                SAMPLE_RATE

            ).squeeze(0)
        )


    n = waveform.numel()


    if n > NUM_SAMPLES:

        max_start = (
            n - NUM_SAMPLES
        )

        start = (
            max_start // 2
        )


        waveform = waveform[
            start:
            start + NUM_SAMPLES
        ]


    elif n < NUM_SAMPLES:

        missing = (
            NUM_SAMPLES - n
        )

        left = (
            missing // 2
        )

        right = (
            missing - left
        )


        waveform = F.pad(

            waveform,

            (
                left,
                right
            )
        )


    return waveform


# =====================================================================
# LOG-MEL FRONTEND
# =====================================================================

class LogMelFrontEnd(nn.Module):

    def __init__(self):

        super().__init__()


        self.mel = (
            torchaudio.transforms
            .MelSpectrogram(

                sample_rate=
                    SAMPLE_RATE,

                n_fft=
                    N_FFT,

                hop_length=
                    HOP_LENGTH,

                n_mels=
                    N_MELS,

                power=
                    2.0,
            )
        )


        self.db = (
            torchaudio.transforms
            .AmplitudeToDB(

                stype="power",

                top_db=80
            )
        )


    def forward(
        self,
        waveform
    ):

        x = self.mel(
            waveform
        )

        x = self.db(
            x
        )


        # Normalización por muestra
        mean = x.mean(

            dim=(-2, -1),

            keepdim=True
        )


        std = (
            x.std(

                dim=(-2, -1),

                keepdim=True

            ).clamp_min(
                1e-6
            )
        )


        x = (
            x - mean
        ) / std


        return x.unsqueeze(1)


# =====================================================================
# CNN
# =====================================================================

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.frontend = (
            LogMelFrontEnd()
        )


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(2),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                1
            ),
        )


    def forward(
        self,
        waveform
    ):

        x = self.frontend(
            waveform
        )

        x = self.features(
            x
        )

        return (
            self.classifier(
                x
            )
            .squeeze(1)
        )


# =====================================================================
# METRICS
# =====================================================================

def confusion_metrics(
    labels,
    probs,
    threshold
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    probs = np.asarray(
        probs,
        dtype=float
    )


    preds = (
        probs >= threshold
    ).astype(int)


    tp = int(np.sum(
        (preds == 1)
        & (labels == 1)
    ))

    tn = int(np.sum(
        (preds == 0)
        & (labels == 0)
    ))

    fp = int(np.sum(
        (preds == 1)
        & (labels == 0)
    ))

    fn = int(np.sum(
        (preds == 0)
        & (labels == 1)
    ))


    accuracy = (
        (tp + tn)
        / len(labels)
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


    balanced_accuracy = (
        recall + specificity
    ) / 2.0


    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )


    fnr = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )


    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "fpr":
            fpr,

        "fnr":
            fnr,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,
    }


def roc_auc(
    labels,
    probs
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    probs = np.asarray(
        probs,
        dtype=float
    )


    positive = probs[
        labels == 1
    ]

    negative = probs[
        labels == 0
    ]


    score = 0.0


    for p in positive:

        score += np.sum(
            p > negative
        )

        score += (
            0.5
            * np.sum(
                p == negative
            )
        )


    return float(
        score
        / (
            len(positive)
            * len(negative)
        )
    )


def eer(
    labels,
    probs
):

    thresholds = np.unique(

        np.concatenate(
            [
                [-1e-9],
                probs,
                [1.0 + 1e-9],
            ]
        )
    )


    best = None


    for threshold in thresholds:

        metrics = confusion_metrics(

            labels,

            probs,

            float(threshold)
        )


        candidate = (

            abs(
                metrics["fpr"]
                - metrics["fnr"]
            ),

            (
                metrics["fpr"]
                + metrics["fnr"]
            ) / 2,

            float(threshold),
        )


        if (
            best is None
            or candidate[0] < best[0]
        ):

            best = candidate


    return (
        float(best[1]),
        float(best[2])
    )


# =====================================================================
# GENERATOR BREAKDOWN
# =====================================================================

def by_generator(
    predictions
):

    groups = defaultdict(
        list
    )


    for row in predictions:

        groups[
            row["generator"]
        ].append(
            row
        )


    order = [

        "VoxPopuli",
        "RVC",
        "Qwen3-TTS",
        "Fun-CosyVoice3",
        "Confucius4-TTS",
        "OpenVoice V2",
        "VoxCPM2",
    ]


    result = []


    for generator in order:

        rows = groups[
            generator
        ]


        label = int(
            rows[0]["label"]
        )


        probs = np.asarray(

            [
                float(
                    row["prob_fake"]
                )
                for row in rows
            ]
        )


        preds = np.asarray(

            [
                int(
                    row["pred_label"]
                )
                for row in rows
            ]
        )


        labels = np.full(
            len(rows),
            label
        )


        correct = int(
            np.sum(
                preds == labels
            )
        )


        if label == 0:

            metric_name = (
                "specificity"
            )

            metric_value = (
                np.sum(preds == 0)
                / len(rows)
            )

        else:

            metric_name = (
                "fake_recall"
            )

            metric_value = (
                np.sum(preds == 1)
                / len(rows)
            )


        result.append({

            "generator":
                generator,

            "true_label":
                label,

            "n":
                len(rows),

            "correct":
                correct,

            "errors":
                len(rows)
                - correct,

            "accuracy":
                correct
                / len(rows),

            "primary_metric":
                metric_name,

            "primary_value":
                float(
                    metric_value
                ),

            "prob_fake_mean":
                float(
                    probs.mean()
                ),

            "prob_fake_median":
                float(
                    np.median(
                        probs
                    )
                ),

            "prob_fake_min":
                float(
                    probs.min()
                ),

            "prob_fake_max":
                float(
                    probs.max()
                ),
        })


    return result


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()


    if not CHECKPOINT.exists():

        fail(
            f"No existe checkpoint:\n"
            f"{CHECKPOINT}"
        )


    if (
        (
            PREDICTIONS_CSV.exists()
            or METRICS_JSON.exists()
            or BY_GENERATOR_CSV.exists()
        )
        and not args.force
    ):

        fail(
            "Ya existen resultados CNN.\n"
            "No se sobrescriben."
        )


    rows = verify_holdout()


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    print()
    print("=" * 76)
    print("CARGANDO BASELINE LOG-MEL CNN")
    print("=" * 76)

    print(
        f"Checkpoint:\n"
        f"  {CHECKPOINT}"
    )

    print(
        f"SHA256     : "
        f"{sha256(CHECKPOINT)}"
    )

    print(
        f"Device     : {device}"
    )

    print(
        f"Threshold  : "
        f"{FIXED_THRESHOLD}"
    )


    checkpoint = torch.load(

        CHECKPOINT,

        map_location="cpu",

        weights_only=False
    )


    model = (
        BaselineCNN()
        .to(device)
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ],

        strict=True
    )


    model.eval()


    predictions = []


    print()
    print("=" * 76)
    print("CNN — EXTERNAL FINAL HOLDOUT")
    print("=" * 76)


    with torch.no_grad():

        for start in range(
            0,
            len(rows),
            args.batch_size
        ):

            batch_rows = rows[
                start:
                start + args.batch_size
            ]


            waveforms = []


            for row in batch_rows:

                path = (
                    ROOT
                    / Path(
                        row[
                            "relative_path"
                        ]
                    )
                )


                waveforms.append(
                    load_audio(
                        path
                    )
                )


            x = torch.stack(
                waveforms
            ).to(
                device
            )


            logits = model(
                x
            )


            probs = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )


            for row, prob in zip(
                batch_rows,
                probs
            ):

                prob = float(
                    prob
                )

                label = int(
                    row["label"]
                )


                pred = (
                    1
                    if prob
                    >= FIXED_THRESHOLD
                    else 0
                )


                if (
                    label == 0
                    and pred == 1
                ):

                    error_type = "FP"

                elif (
                    label == 1
                    and pred == 0
                ):

                    error_type = "FN"

                else:

                    error_type = ""


                predictions.append({

                    "id":
                        row["id"],

                    "relative_path":
                        row[
                            "relative_path"
                        ],

                    "file":
                        row["file"],

                    "label":
                        label,

                    "class":
                        row["class"],

                    "generator":
                        row[
                            "generator"
                        ],

                    "speaker":
                        row[
                            "speaker"
                        ],

                    "prob_fake":
                        prob,

                    "threshold":
                        FIXED_THRESHOLD,

                    "pred_label":
                        pred,

                    "pred_class":
                        (
                            "fake"
                            if pred
                            else "real"
                        ),

                    "correct":
                        int(
                            pred == label
                        ),

                    "error_type":
                        error_type,
                })


            done = min(
                start
                + args.batch_size,
                len(rows)
            )


            print(
                f"\rProcesados: "
                f"{done}/{len(rows)}",
                end="",
                flush=True
            )


    print()


    labels = np.asarray(

        [
            int(
                row["label"]
            )
            for row
            in predictions
        ]
    )


    probs = np.asarray(

        [
            float(
                row["prob_fake"]
            )
            for row
            in predictions
        ]
    )


    metrics = confusion_metrics(

        labels,

        probs,

        FIXED_THRESHOLD
    )


    metrics[
        "roc_auc"
    ] = roc_auc(
        labels,
        probs
    )


    (
        eer_value,
        eer_threshold

    ) = eer(
        labels,
        probs
    )


    metrics[
        "eer"
    ] = eer_value

    metrics[
        "eer_threshold"
    ] = eer_threshold


    generator_rows = (
        by_generator(
            predictions
        )
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    write_csv(
        PREDICTIONS_CSV,
        predictions
    )


    write_csv(
        BY_GENERATOR_CSV,
        generator_rows
    )


    result = {

        "evaluation":
            "external_final_holdout",

        "model":
            "baseline_logmel_cnn",

        "threshold":
            FIXED_THRESHOLD,

        "threshold_policy":
            (
                "Original fixed threshold "
                "used by the baseline CNN. "
                "No recalibration on external holdout."
            ),

        "dataset_fingerprint":
            EXPECTED_FINGERPRINT,

        "checkpoint": {

            "path":
                str(
                    CHECKPOINT
                ),

            "sha256":
                sha256(
                    CHECKPOINT
                ),

            "epoch":
                checkpoint.get(
                    "epoch"
                ),

            "val_f1":
                checkpoint.get(
                    "val_f1"
                ),

            "config":
                checkpoint.get(
                    "config"
                ),
        },

        "global_metrics":
            metrics,

        "by_generator":
            generator_rows,
    }


    with METRICS_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )


    print()
    print("=" * 76)
    print("CNN — EXTERNAL FINAL HOLDOUT")
    print("=" * 76)

    print(
        f"Audios          : "
        f"{len(predictions)}"
    )

    print(
        f"Threshold FIJO  : "
        f"{FIXED_THRESHOLD:.6f}"
    )

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc.   : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision       : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall fake     : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity     : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1              : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"EER             : "
        f"{metrics['eer']:.4f}"
    )


    print()

    print(
        f"TN={metrics['tn']}  "
        f"FP={metrics['fp']}  "
        f"FN={metrics['fn']}  "
        f"TP={metrics['tp']}"
    )


    print()
    print(
        "RESULTADO POR ORIGEN"
    )

    print("-" * 76)


    for row in generator_rows:

        print(
            f"{row['generator']:<20} "
            f"N={row['n']:>3}  "
            f"{row['primary_metric']}="
            f"{row['primary_value']:.4f}  "
            f"p_fake_mean="
            f"{row['prob_fake_mean']:.4f}"
        )


    print()
    print(
        f"Predicciones : "
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"Métricas     : "
        f"{METRICS_JSON}"
    )

    print(
        f"Por generador: "
        f"{BY_GENERATOR_CSV}"
    )

    print()
    print(
        "HOLDOUT NO MODIFICADO."
    )

    print("=" * 76)


if __name__ == "__main__":
    main()