from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WavLMModel


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection"
)

HOLDOUT_CSV = (
    ROOT
    / "external_final_holdout"
    / "metadata_external_final.csv"
)

HOLDOUT_MANIFEST = (
    ROOT
    / "external_final_holdout"
    / "freeze_manifest_external_final.json"
)

EXPECTED_FINGERPRINT = (
    "3fed1b260a404af75b21aab8a57f1cf"
    "2dd0dbc8967d2695b45780095008f6021"
)

# Reutilizamos las copias 16 kHz creadas durante
# la primera evaluación WavLM.
PREPROCESS_MANIFEST = (
    ROOT
    / "external_results"
    / "wavlm_final"
    / "preprocessing_manifest.csv"
)

MODEL_NAME = "microsoft/wavlm-base-plus"

TARGET_SR = 16000
SECONDS = 4
MAX_SAMPLES = TARGET_SR * SECONDS

SEEDS = [
    42,
    123,
    2026,
]

EXPERIMENT_ROOT = (
    ROOT
    / "experiments"
    / "wavlm_domain_adapt_real"
)

OUTPUT_ROOT = (
    ROOT
    / "external_results"
    / "wavlm_domain_adapt_final"
)

SUMMARY_RUNS = (
    OUTPUT_ROOT
    / "summary_all_runs.csv"
)

SUMMARY_GLOBAL = (
    OUTPUT_ROOT
    / "summary_mean_std.json"
)

SUMMARY_GENERATORS = (
    OUTPUT_ROOT
    / "summary_by_generator.csv"
)


# ============================================================
# UTILS
# ============================================================

def fail(message):

    print()
    print("=" * 78)
    print("ERROR")
    print("=" * 78)
    print(message)
    print("=" * 78)

    sys.exit(1)


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):

    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# VERIFY HOLDOUT
# ============================================================

def load_holdout():

    if not HOLDOUT_CSV.exists():
        fail(
            f"No existe:\n{HOLDOUT_CSV}"
        )

    if not HOLDOUT_MANIFEST.exists():
        fail(
            f"No existe:\n{HOLDOUT_MANIFEST}"
        )

    if not PREPROCESS_MANIFEST.exists():
        fail(
            "No encuentro el manifest del "
            "preprocesado 16 kHz:\n"
            f"{PREPROCESS_MANIFEST}"
        )


    with HOLDOUT_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        freeze = json.load(f)


    fingerprint = freeze.get(
        "dataset_fingerprint_sha256"
    )

    if fingerprint != EXPECTED_FINGERPRINT:

        fail(
            "Fingerprint del holdout diferente.\n\n"
            f"Esperado:\n{EXPECTED_FINGERPRINT}\n\n"
            f"Actual:\n{fingerprint}"
        )


    rows = read_csv(
        HOLDOUT_CSV
    )

    if len(rows) != 220:

        fail(
            f"Esperaba 220 audios; hay {len(rows)}."
        )


    # --------------------------------------------------------
    # VERIFY ORIGINAL WAVS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("VERIFICANDO HOLDOUT")
    print("=" * 78)


    for i, row in enumerate(
        rows,
        start=1,
    ):

        path = (
            ROOT
            / Path(
                row["relative_path"]
            )
        )

        if not path.exists():

            fail(
                f"Falta WAV:\n{path}"
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
                f"{path}"
            )


        if (
            i % 25 == 0
            or i == len(rows)
        ):

            print(
                f"\rOriginales verificados: "
                f"{i}/{len(rows)}",
                end="",
                flush=True,
            )


    print()


    # --------------------------------------------------------
    # MAP PREPROCESSED COPIES
    # --------------------------------------------------------

    prep_rows = read_csv(
        PREPROCESS_MANIFEST
    )


    prep_by_source = {

        row["source_relative_path"]:
            row

        for row in prep_rows
    }


    for row in rows:

        relative = row[
            "relative_path"
        ]

        prep = prep_by_source.get(
            relative
        )


        if prep is None:

            fail(
                "No encuentro copia 16 kHz para:\n"
                f"{relative}"
            )


        if (
            prep["source_sha256"]
            .lower()
            != row["sha256"]
            .lower()
        ):

            fail(
                "El manifest de preprocesado "
                "no corresponde al WAV congelado:\n"
                f"{relative}"
            )


        normalized = Path(
            prep["normalized_path"]
        )


        if not normalized.exists():

            fail(
                f"No existe:\n{normalized}"
            )


        if (
            sha256(normalized).lower()
            != prep["normalized_sha256"].lower()
        ):

            fail(
                "Copia normalizada modificada:\n"
                f"{normalized}"
            )


        info = sf.info(
            str(normalized)
        )


        if (
            info.samplerate != 16000
            or info.channels != 1
        ):

            fail(
                "Preprocesado inválido:\n"
                f"{normalized}"
            )


        row[
            "_normalized_path"
        ] = str(
            normalized
        )


    print(
        "Fingerprint :",
        fingerprint,
    )

    print(
        "Integridad  : OK"
    )


    return rows


# ============================================================
# MODEL
# ============================================================

class WavLMClassifier(nn.Module):

    def __init__(self):

        super().__init__()


        self.wavlm = (
            WavLMModel.from_pretrained(
                MODEL_NAME,
                local_files_only=True,
            )
        )


        hidden = (
            self.wavlm
            .config
            .hidden_size
        )


        self.head = nn.Sequential(

            nn.LayerNorm(
                hidden
            ),

            nn.Linear(
                hidden,
                256,
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                1,
            ),
        )


    def forward(
        self,
        input_values,
        attention_mask,
    ):

        out = self.wavlm(

            input_values=
                input_values,

            attention_mask=
                attention_mask,
        )


        h = (
            out
            .last_hidden_state
        )


        feature_mask = (
            self.wavlm
            ._get_feature_vector_attention_mask(
                h.shape[1],
                attention_mask,
            )
            .to(h.device)
        )


        mask = (
            feature_mask
            .unsqueeze(-1)
            .to(h.dtype)
        )


        pooled = (
            h * mask
        ).sum(dim=1)


        pooled = pooled / (
            mask
            .sum(dim=1)
            .clamp_min(1.0)
        )


        return (
            self.head(
                pooled
            )
            .squeeze(-1)
        )


# ============================================================
# TEST PREPROCESSING
#
# EXACTO DEL SCRIPT domain_adapt_real:
#
# >4 s -> crop central
# <4 s -> zero padding A LA DERECHA
# ============================================================

def load_audio(path):

    x, sr = sf.read(

        str(path),

        dtype="float32",

        always_2d=False,
    )


    if x.ndim == 2:

        x = x.mean(
            axis=1
        )


    if sr != TARGET_SR:

        raise RuntimeError(
            f"{path}: {sr} Hz; "
            f"esperaba {TARGET_SR}."
        )


    x = np.asarray(
        x,
        dtype=np.float32,
    )


    if len(x) > MAX_SAMPLES:

        start = (
            len(x)
            - MAX_SAMPLES
        ) // 2


        x = x[
            start:
            start + MAX_SAMPLES
        ]


    valid_len = min(
        len(x),
        MAX_SAMPLES,
    )


    if len(x) < MAX_SAMPLES:

        x = np.pad(

            x,

            (
                0,
                MAX_SAMPLES
                - len(x),
            ),

            mode="constant",
        )


    mask = np.zeros(
        MAX_SAMPLES,
        dtype=np.int64,
    )


    mask[
        :valid_len
    ] = 1


    return (
        x.astype(
            np.float32
        ),
        mask,
    )


# ============================================================
# METRICS
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


    tp = int(np.sum(
        (pred == 1)
        & (labels == 1)
    ))

    tn = int(np.sum(
        (pred == 0)
        & (labels == 0)
    ))

    fp = int(np.sum(
        (pred == 1)
        & (labels == 0)
    ))

    fn = int(np.sum(
        (pred == 0)
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


    positives = probs[
        labels == 1
    ]

    negatives = probs[
        labels == 0
    ]


    result = 0.0


    for p in positives:

        result += float(
            np.sum(
                p > negatives
            )
        )

        result += (
            0.5
            * float(
                np.sum(
                    p == negatives
                )
            )
        )


    return (
        result
        / (
            len(positives)
            * len(negatives)
        )
    )


def eer(
    labels,
    probs,
):

    thresholds = np.unique(

        np.concatenate(
            [
                np.array([-1e-9]),
                probs,
                np.array([1.0 + 1e-9]),
            ]
        )
    )


    best = None


    for threshold in thresholds:

        m = confusion_metrics(

            labels,

            probs,

            float(threshold),
        )


        candidate = (

            abs(
                m["fpr"]
                - m["fnr"]
            ),

            (
                m["fpr"]
                + m["fnr"]
            ) / 2.0,

            float(threshold),
        )


        if (
            best is None
            or candidate[0]
            < best[0]
        ):

            best = candidate


    return (
        float(best[1]),
        float(best[2]),
    )


# ============================================================
# ONE SEED
# ============================================================

@torch.inference_mode()
def evaluate_seed(
    seed,
    rows,
    device,
):

    run_dir = (

        EXPERIMENT_ROOT
        / f"seed_{seed}"

    )


    checkpoint_path = (
        run_dir
        / "best_model.pt"
    )


    metrics_path = (
        run_dir
        / "metrics.json"
    )


    if not checkpoint_path.exists():

        fail(
            f"No existe:\n"
            f"{checkpoint_path}"
        )


    if not metrics_path.exists():

        fail(
            f"No existe:\n"
            f"{metrics_path}"
        )


    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        internal_metrics = json.load(f)


    threshold = float(
        internal_metrics[
            "calibrated_threshold"
        ]
    )


    print()
    print("=" * 78)

    print(
        f"DOMAIN ADAPT REAL | "
        f"SEED={seed}"
    )

    print("=" * 78)

    print(
        f"Threshold VAL : "
        f"{threshold:.9f}"
    )


    model = (
        WavLMClassifier()
        .to(device)
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu",

        weights_only=False,
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ],

        strict=True,
    )


    model.eval()


    predictions = []

    batch_size = 4


    for start in range(
        0,
        len(rows),
        batch_size,
    ):

        batch = rows[
            start:
            start + batch_size
        ]


        xs = []
        masks = []


        for row in batch:

            x, mask = load_audio(

                Path(
                    row[
                        "_normalized_path"
                    ]
                )

            )


            xs.append(x)
            masks.append(mask)


        x_tensor = torch.tensor(

            np.stack(xs),

            dtype=torch.float32,

            device=device,
        )


        mask_tensor = torch.tensor(

            np.stack(masks),

            dtype=torch.long,

            device=device,
        )


        if device.type == "cuda":

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16,

            ):

                logits = model(
                    x_tensor,
                    mask_tensor,
                )

        else:

            logits = model(
                x_tensor,
                mask_tensor,
            )


        probs = (
            torch.sigmoid(
                logits.float()
            )
            .cpu()
            .numpy()
        )


        for row, prob in zip(
            batch,
            probs,
        ):

            prob = float(prob)

            label = int(
                row["label"]
            )

            pred = int(
                prob >= threshold
            )


            predictions.append({

                "id":
                    row["id"],

                "relative_path":
                    row[
                        "relative_path"
                    ],

                "generator":
                    row[
                        "generator"
                    ],

                "speaker":
                    row[
                        "speaker"
                    ],

                "label":
                    label,

                "prob_fake":
                    prob,

                "threshold":
                    threshold,

                "pred_label":
                    pred,

                "correct":
                    int(
                        pred == label
                    ),
            })


        done = min(
            start + batch_size,
            len(rows),
        )


        print(
            f"\rProcesados: "
            f"{done}/{len(rows)}",
            end="",
            flush=True,
        )


    print()


    labels = np.asarray(
        [
            int(r["label"])
            for r in predictions
        ]
    )


    probs = np.asarray(
        [
            float(r["prob_fake"])
            for r in predictions
        ]
    )


    metrics = confusion_metrics(

        labels,

        probs,

        threshold,
    )


    metrics[
        "roc_auc"
    ] = roc_auc(
        labels,
        probs,
    )


    eer_value, eer_threshold = (
        eer(
            labels,
            probs,
        )
    )


    metrics[
        "eer"
    ] = eer_value

    metrics[
        "eer_threshold"
    ] = eer_threshold


    # --------------------------------------------------------
    # BY GENERATOR
    # --------------------------------------------------------

    groups = defaultdict(
        list
    )


    for row in predictions:

        groups[
            row["generator"]
        ].append(row)


    generator_rows = []


    for generator in [

        "VoxPopuli",
        "RVC",
        "Qwen3-TTS",
        "Fun-CosyVoice3",
        "Confucius4-TTS",
        "OpenVoice V2",
        "VoxCPM2",

    ]:

        items = groups[
            generator
        ]


        true_label = int(
            items[0]["label"]
        )


        correct = sum(

            int(item["correct"])

            for item in items
        )


        probability_mean = float(

            np.mean(
                [
                    item["prob_fake"]
                    for item in items
                ]
            )
        )


        if true_label == 0:

            primary_name = (
                "specificity"
            )

            primary_value = (
                sum(
                    item["pred_label"] == 0
                    for item in items
                )
                / len(items)
            )

        else:

            primary_name = (
                "fake_recall"
            )

            primary_value = (
                sum(
                    item["pred_label"] == 1
                    for item in items
                )
                / len(items)
            )


        generator_rows.append({

            "seed":
                seed,

            "generator":
                generator,

            "n":
                len(items),

            "primary_metric":
                primary_name,

            "primary_value":
                primary_value,

            "correct":
                correct,

            "errors":
                len(items) - correct,

            "prob_fake_mean":
                probability_mean,
        })


    # --------------------------------------------------------
    # SAVE RUN
    # --------------------------------------------------------

    result_dir = (
        OUTPUT_ROOT
        / f"seed_{seed}"
    )


    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_csv(

        result_dir
        / "predictions.csv",

        predictions,
    )


    write_csv(

        result_dir
        / "by_generator.csv",

        generator_rows,
    )


    run_result = {

        "seed":
            seed,

        "threshold":
            threshold,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            sha256(
                checkpoint_path
            ),

        **metrics,
    }


    with (
        result_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    print(
        f"Accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc. : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Recall fake   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity   : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1            : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC       : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"EER           : "
        f"{metrics['eer']:.4f}"
    )

    print(
        f"TN={metrics['tn']}  "
        f"FP={metrics['fp']}  "
        f"FN={metrics['fn']}  "
        f"TP={metrics['tp']}"
    )


    return (
        run_result,
        generator_rows,
    )


# ============================================================
# MEAN / STD
# ============================================================

def mean_std(values):

    values = [
        float(v)
        for v in values
    ]


    mean = statistics.mean(
        values
    )


    std = (

        statistics.stdev(
            values
        )

        if len(values) > 1

        else 0.0
    )


    return mean, std


# ============================================================
# MAIN
# ============================================================

def main():

    rows = load_holdout()


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    print()
    print("=" * 78)
    print(
        "WAVLM DOMAIN ADAPT REAL — "
        "FINAL EXTERNAL HOLDOUT"
    )
    print("=" * 78)

    print(
        f"Device : {device}"
    )

    if device.type == "cuda":

        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )


    all_runs = []
    all_generator_rows = []


    for seed in SEEDS:

        run_result, generator_rows = (
            evaluate_seed(
                seed,
                rows,
                device,
            )
        )


        all_runs.append(
            run_result
        )

        all_generator_rows.extend(
            generator_rows
        )


        if device.type == "cuda":

            torch.cuda.empty_cache()


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_csv(
        SUMMARY_RUNS,
        all_runs,
    )


    # ========================================================
    # GLOBAL MEAN ± STD
    # ========================================================

    metrics_to_aggregate = [

        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "eer",
        "threshold",
    ]


    summary = {

        "model":
            "WavLM domain_adapt_real",

        "seeds":
            SEEDS,

        "dataset_fingerprint":
            EXPECTED_FINGERPRINT,

        "metrics": {},
    }


    for metric in metrics_to_aggregate:

        mean, std = mean_std(

            [
                row[metric]
                for row in all_runs
            ]
        )


        summary[
            "metrics"
        ][metric] = {

            "mean":
                mean,

            "std":
                std,
        }


    with SUMMARY_GLOBAL.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # GENERATOR MEAN ± STD
    # ========================================================

    generator_summary = []


    for generator in [

        "VoxPopuli",
        "RVC",
        "Qwen3-TTS",
        "Fun-CosyVoice3",
        "Confucius4-TTS",
        "OpenVoice V2",
        "VoxCPM2",

    ]:

        group = [

            row

            for row
            in all_generator_rows

            if row[
                "generator"
            ] == generator
        ]


        primary_mean, primary_std = (
            mean_std(
                [
                    row[
                        "primary_value"
                    ]
                    for row in group
                ]
            )
        )


        prob_mean, prob_std = (
            mean_std(
                [
                    row[
                        "prob_fake_mean"
                    ]
                    for row in group
                ]
            )
        )


        generator_summary.append({

            "generator":
                generator,

            "metric":
                group[0][
                    "primary_metric"
                ],

            "mean":
                primary_mean,

            "std":
                primary_std,

            "prob_fake_mean":
                prob_mean,

            "prob_fake_std":
                prob_std,
        })


    write_csv(
        SUMMARY_GENERATORS,
        generator_summary,
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)

    print(
        "DOMAIN ADAPT REAL — "
        "MEDIA ± STD"
    )

    print("=" * 78)


    for metric in [

        "accuracy",
        "balanced_accuracy",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "eer",

    ]:

        values = summary[
            "metrics"
        ][metric]


        print(
            f"{metric:<18}: "
            f"{values['mean']:.4f} "
            f"± {values['std']:.4f}"
        )


    print()
    print(
        "POR GENERADOR"
    )

    print("-" * 78)


    for row in generator_summary:

        print(
            f"{row['generator']:<20} "
            f"{row['metric']}="
            f"{row['mean']:.4f} "
            f"± {row['std']:.4f}"
        )


    print()
    print(
        f"Runs          : "
        f"{SUMMARY_RUNS}"
    )

    print(
        f"Resumen       : "
        f"{SUMMARY_GLOBAL}"
    )

    print(
        f"Por generador : "
        f"{SUMMARY_GENERATORS}"
    )

    print()
    print(
        "HOLDOUT NO MODIFICADO."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()