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

PREPROCESS_MANIFEST = (
    ROOT
    / "external_results"
    / "wavlm_final"
    / "preprocessing_manifest.csv"
)

INITIAL_WAVLM_METRICS = (
    ROOT
    / "external_results"
    / "wavlm_final"
    / "wavlm_metrics.json"
)

EXPERIMENT_ROOT = (
    ROOT
    / "experiments"
    / "wavlm_domain_robust"
)

OUTPUT_ROOT = (
    ROOT
    / "external_results"
    / "wavlm_robust_suite_final"
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

MODES = [
    "control",
    "robust",
    "gate_only",
    "channel_mix",
]

GENERATORS = [
    "VoxPopuli",
    "RVC",
    "Qwen3-TTS",
    "Fun-CosyVoice3",
    "Confucius4-TTS",
    "OpenVoice V2",
    "VoxCPM2",
]


# ============================================================
# UTILS
# ============================================================

def fail(message):

    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    print("=" * 80)

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


def mean_std(values):

    values = [
        float(x)
        for x in values
    ]

    mean = statistics.mean(
        values
    )

    std = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return mean, std


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
            f"No existe:\n{PREPROCESS_MANIFEST}"
        )


    with HOLDOUT_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)


    fingerprint = manifest.get(
        "dataset_fingerprint_sha256"
    )


    if fingerprint != EXPECTED_FINGERPRINT:

        fail(
            "Fingerprint incorrecto.\n\n"
            f"Esperado:\n{EXPECTED_FINGERPRINT}\n"
            f"Actual:\n{fingerprint}"
        )


    rows = read_csv(
        HOLDOUT_CSV
    )


    if len(rows) != 220:

        fail(
            f"Hay {len(rows)} filas; esperaba 220."
        )


    print()
    print("=" * 80)
    print("VERIFICANDO HOLDOUT EXTERNO")
    print("=" * 80)


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
                f"Falta:\n{path}"
            )


        if (
            sha256(path).lower()
            != row["sha256"].lower()
        ):

            fail(
                f"WAV MODIFICADO:\n{path}"
            )


        if (
            i % 25 == 0
            or i == 220
        ):

            print(
                f"\rOriginales verificados: {i}/220",
                end="",
                flush=True,
            )


    print()


    # --------------------------------------------------------
    # Preprocessed 16 kHz copies
    # --------------------------------------------------------

    prep_rows = read_csv(
        PREPROCESS_MANIFEST
    )


    prep_map = {

        row["source_relative_path"]:
            row

        for row in prep_rows
    }


    for row in rows:

        relative = (
            row["relative_path"]
        )


        prep = prep_map.get(
            relative
        )


        if prep is None:

            fail(
                "No existe preprocesado para:\n"
                f"{relative}"
            )


        if (
            prep["source_sha256"].lower()
            != row["sha256"].lower()
        ):

            fail(
                "El preprocesado no corresponde "
                "al WAV congelado:\n"
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
                "Preprocesado modificado:\n"
                f"{normalized}"
            )


        info = sf.info(
            str(normalized)
        )


        if (
            info.samplerate != TARGET_SR
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
# ORIGINAL WAVLM CHECKPOINT HASH
# ============================================================

def get_initial_checkpoint_hash():

    if not INITIAL_WAVLM_METRICS.exists():
        return None


    try:

        with INITIAL_WAVLM_METRICS.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)


        return (
            data
            .get("checkpoint", {})
            .get("sha256")
        )

    except Exception:

        return None


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

            input_values=input_values,

            attention_mask=attention_mask,
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
# Igual a robust / gate_only / channel_mix:
#
# >4 s  -> crop central
# <4 s  -> padding a la derecha
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
                MAX_SAMPLES - len(x),
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


    positive = probs[
        labels == 1
    ]

    negative = probs[
        labels == 0
    ]


    if (
        len(positive) == 0
        or len(negative) == 0
    ):

        return float("nan")


    score = 0.0


    for p in positive:

        score += float(
            np.sum(
                p > negative
            )
        )

        score += (
            0.5
            * float(
                np.sum(
                    p == negative
                )
            )
        )


    return score / (
        len(positive)
        * len(negative)
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
        float(best[2]),
    )


# ============================================================
# CHECK MODE AVAILABILITY
# ============================================================

def complete_mode(mode):

    missing = []


    for seed in SEEDS:

        folder = (
            EXPERIMENT_ROOT
            / mode
            / f"seed_{seed}"
        )


        for filename in [
            "best_model.pt",
            "metrics.json",
        ]:

            path = (
                folder
                / filename
            )


            if not path.exists():

                missing.append(
                    str(path)
                )


    return missing


# ============================================================
# RUN ONE CHECKPOINT
# ============================================================

@torch.inference_mode()
def evaluate_one(
    mode,
    seed,
    rows,
    device,
    initial_checkpoint_hash,
):

    run_dir = (

        EXPERIMENT_ROOT
        / mode
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


    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        saved_metrics = json.load(f)


    if "calibrated_threshold" not in saved_metrics:

        fail(
            "metrics.json no contiene "
            "calibrated_threshold:\n"
            f"{metrics_path}"
        )


    threshold = float(
        saved_metrics[
            "calibrated_threshold"
        ]
    )


    checkpoint_hash = sha256(
        checkpoint_path
    )


    duplicate_initial = bool(

        initial_checkpoint_hash

        and checkpoint_hash
        == initial_checkpoint_hash
    )


    print()
    print("=" * 80)

    print(
        f"MODE={mode} | SEED={seed}"
    )

    print("=" * 80)

    print(
        f"Threshold VAL : "
        f"{threshold:.9f}"
    )

    print(
        f"Checkpoint    : "
        f"{checkpoint_hash}"
    )

    print(
        "Mismo checkpoint que WavLM inicial : "
        f"{duplicate_initial}"
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


        x_batch = []
        mask_batch = []


        for row in batch:

            x, mask = load_audio(

                Path(
                    row[
                        "_normalized_path"
                    ]
                )

            )


            x_batch.append(x)
            mask_batch.append(mask)


        x_tensor = torch.tensor(

            np.stack(
                x_batch
            ),

            dtype=torch.float32,

            device=device,
        )


        mask_tensor = torch.tensor(

            np.stack(
                mask_batch
            ),

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


        probabilities = (

            torch.sigmoid(
                logits.float()
            )
            .cpu()
            .numpy()
        )


        for row, probability in zip(
            batch,
            probabilities,
        ):

            probability = float(
                probability
            )


            label = int(
                row["label"]
            )


            pred = int(
                probability >= threshold
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
                    probability,

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
            int(row["label"])
            for row in predictions
        ]
    )


    probs = np.asarray(

        [
            float(row["prob_fake"])
            for row in predictions
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
    # GENERATOR RESULTS
    # --------------------------------------------------------

    grouped = defaultdict(
        list
    )


    for row in predictions:

        grouped[
            row["generator"]
        ].append(row)


    generator_results = []


    for generator in GENERATORS:

        items = grouped[
            generator
        ]


        true_label = int(
            items[0]["label"]
        )


        if true_label == 0:

            metric_name = (
                "specificity"
            )

            value = (

                sum(
                    row["pred_label"] == 0
                    for row in items
                )

                / len(items)
            )

        else:

            metric_name = (
                "fake_recall"
            )

            value = (

                sum(
                    row["pred_label"] == 1
                    for row in items
                )

                / len(items)
            )


        generator_results.append({

            "mode":
                mode,

            "seed":
                seed,

            "generator":
                generator,

            "metric":
                metric_name,

            "value":
                value,

            "prob_fake_mean":
                float(
                    np.mean(
                        [
                            row["prob_fake"]
                            for row in items
                        ]
                    )
                ),
        })


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result_dir = (

        OUTPUT_ROOT
        / mode
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

        generator_results,
    )


    run_result = {

        "mode":
            mode,

        "seed":
            seed,

        "threshold":
            threshold,

        "same_as_initial_wavlm":
            duplicate_initial,

        "checkpoint_sha256":
            checkpoint_hash,

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


    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()


    return (
        run_result,
        generator_results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rows = load_holdout()


    initial_checkpoint_hash = (
        get_initial_checkpoint_hash()
    )


    if initial_checkpoint_hash:

        print()
        print(
            "WavLM inicial checkpoint SHA256:"
        )

        print(
            initial_checkpoint_hash
        )


    # --------------------------------------------------------
    # Determine complete experiment modes
    # --------------------------------------------------------

    available_modes = []


    print()
    print("=" * 80)
    print("COMPROBANDO EXPERIMENTOS")
    print("=" * 80)


    for mode in MODES:

        missing = complete_mode(
            mode
        )


        if missing:

            print(
                f"[SKIP] {mode}: "
                "faltan checkpoints/metrics"
            )

            for path in missing:

                print(
                    f"       {path}"
                )

        else:

            available_modes.append(
                mode
            )

            print(
                f"[OK]   {mode}: "
                "3 semillas completas"
            )


    if not available_modes:

        fail(
            "No hay ningún modo completo."
        )


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    print()
    print("=" * 80)

    print(
        "WAVLM ROBUST SUITE — "
        "EXTERNAL FINAL HOLDOUT"
    )

    print("=" * 80)

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


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for mode in available_modes:

        for seed in SEEDS:

            run, generator_rows = (
                evaluate_one(

                    mode,

                    seed,

                    rows,

                    device,

                    initial_checkpoint_hash,
                )
            )


            all_runs.append(
                run
            )

            all_generator_rows.extend(
                generator_rows
            )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_csv(

        OUTPUT_ROOT
        / "summary_all_runs.csv",

        all_runs,
    )


    # ========================================================
    # MODE SUMMARY
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


    mode_summary = []


    for mode in available_modes:

        group = [

            row

            for row in all_runs

            if row["mode"]
            == mode
        ]


        summary_row = {

            "mode":
                mode,

            "n_seeds":
                len(group),
        }


        for metric in metrics_to_aggregate:

            mean, std = mean_std(

                [
                    row[metric]
                    for row in group
                ]
            )


            summary_row[
                f"{metric}_mean"
            ] = mean

            summary_row[
                f"{metric}_std"
            ] = std


        summary_row[
            "same_as_initial_wavlm_count"
        ] = sum(

            bool(
                row[
                    "same_as_initial_wavlm"
                ]
            )

            for row in group
        )


        mode_summary.append(
            summary_row
        )


    write_csv(

        OUTPUT_ROOT
        / "summary_by_mode.csv",

        mode_summary,
    )


    # ========================================================
    # GENERATOR SUMMARY
    # ========================================================

    generator_summary = []


    for mode in available_modes:

        for generator in GENERATORS:

            group = [

                row

                for row
                in all_generator_rows

                if (
                    row["mode"] == mode
                    and row["generator"]
                    == generator
                )
            ]


            mean, std = mean_std(

                [
                    row["value"]
                    for row in group
                ]
            )


            probability_mean, probability_std = (
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

                "mode":
                    mode,

                "generator":
                    generator,

                "metric":
                    group[0]["metric"],

                "mean":
                    mean,

                "std":
                    std,

                "prob_fake_mean":
                    probability_mean,

                "prob_fake_std":
                    probability_std,
            })


    write_csv(

        OUTPUT_ROOT
        / "summary_by_generator.csv",

        generator_summary,
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)

    print(
        "RESUMEN ROBUST SUITE — "
        "MEDIA ± STD"
    )

    print("=" * 80)


    for row in mode_summary:

        print()
        print(
            f"[{row['mode']}]"
        )

        print(
            "  Accuracy      : "
            f"{row['accuracy_mean']:.4f} "
            f"± {row['accuracy_std']:.4f}"
        )

        print(
            "  Balanced Acc. : "
            f"{row['balanced_accuracy_mean']:.4f} "
            f"± {row['balanced_accuracy_std']:.4f}"
        )

        print(
            "  Recall fake   : "
            f"{row['recall_mean']:.4f} "
            f"± {row['recall_std']:.4f}"
        )

        print(
            "  Specificity   : "
            f"{row['specificity_mean']:.4f} "
            f"± {row['specificity_std']:.4f}"
        )

        print(
            "  F1            : "
            f"{row['f1_mean']:.4f} "
            f"± {row['f1_std']:.4f}"
        )

        print(
            "  ROC-AUC       : "
            f"{row['roc_auc_mean']:.4f} "
            f"± {row['roc_auc_std']:.4f}"
        )

        print(
            "  EER           : "
            f"{row['eer_mean']:.4f} "
            f"± {row['eer_std']:.4f}"
        )

        print(
            "  Threshold     : "
            f"{row['threshold_mean']:.4f} "
            f"± {row['threshold_std']:.4f}"
        )


    print()
    print("=" * 80)
    print("POR GENERADOR")
    print("=" * 80)


    for mode in available_modes:

        print()
        print(
            f"[{mode}]"
        )


        for row in generator_summary:

            if row["mode"] != mode:
                continue


            print(
                f"  {row['generator']:<20} "
                f"{row['metric']}="
                f"{row['mean']:.4f} "
                f"± {row['std']:.4f}"
            )


    print()
    print(
        "Resultados:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "HOLDOUT NO MODIFICADO."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()