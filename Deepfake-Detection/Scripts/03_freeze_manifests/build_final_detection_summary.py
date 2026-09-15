from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


# =====================================================================
# CONFIG
# =====================================================================

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


# ---------------------------------------------------------------------
# WavLM inicial
# ---------------------------------------------------------------------

WAVLM_INITIAL_JSON = (
    ROOT
    / "external_results"
    / "wavlm_final"
    / "wavlm_metrics.json"
)


# ---------------------------------------------------------------------
# CNN baseline
# ---------------------------------------------------------------------

CNN_JSON = (
    ROOT
    / "external_results"
    / "baseline_cnn_final"
    / "cnn_metrics.json"
)


# ---------------------------------------------------------------------
# Domain Adapt Real
# ---------------------------------------------------------------------

DOMAIN_ROOT = (
    ROOT
    / "external_results"
    / "wavlm_domain_adapt_final"
)

DOMAIN_SUMMARY_JSON = (
    DOMAIN_ROOT
    / "summary_mean_std.json"
)

DOMAIN_GENERATORS_CSV = (
    DOMAIN_ROOT
    / "summary_by_generator.csv"
)


# ---------------------------------------------------------------------
# Robust suite
# ---------------------------------------------------------------------

ROBUST_ROOT = (
    ROOT
    / "external_results"
    / "wavlm_robust_suite_final"
)

ROBUST_MODES_CSV = (
    ROBUST_ROOT
    / "summary_by_mode.csv"
)

ROBUST_GENERATORS_CSV = (
    ROBUST_ROOT
    / "summary_by_generator.csv"
)


# ---------------------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "external_results"
    / "FINAL_DETECTION_RESULTS"
)

DETECTORS_CSV = (
    OUTPUT_DIR
    / "detector_summary.csv"
)

GENERATORS_CSV = (
    OUTPUT_DIR
    / "generator_summary.csv"
)

SOURCES_CSV = (
    OUTPUT_DIR
    / "source_files_sha256.csv"
)

REPORT_JSON = (
    OUTPUT_DIR
    / "final_detection_report.json"
)

TABLES_MD = (
    OUTPUT_DIR
    / "tables_for_tfg.md"
)

FREEZE_JSON = (
    OUTPUT_DIR
    / "final_results_manifest.json"
)


SEEDS = [
    42,
    123,
    2026,
]

FAKE_GENERATORS = [
    "RVC",
    "Qwen3-TTS",
    "Fun-CosyVoice3",
    "Confucius4-TTS",
    "OpenVoice V2",
    "VoxCPM2",
]

ALL_GENERATORS = [
    "VoxPopuli",
    *FAKE_GENERATORS,
]

DETECTOR_ORDER = [
    "cnn_baseline",
    "wavlm_initial",
    "wavlm_control",
    "wavlm_robust",
    "wavlm_gate_only",
    "wavlm_channel_mix",
    "wavlm_domain_adapt_real",
]

DISPLAY_NAMES = {
    "cnn_baseline":
        "CNN Log-Mel",

    "wavlm_initial":
        "WavLM inicial",

    "wavlm_control":
        "WavLM Control",

    "wavlm_robust":
        "WavLM Robust",

    "wavlm_gate_only":
        "WavLM Gate-only",

    "wavlm_channel_mix":
        "WavLM Channel-mix",

    "wavlm_domain_adapt_real":
        "WavLM Domain-adapt real",
}


# =====================================================================
# UTILS
# =====================================================================

def fail(message):

    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    print("=" * 80)

    sys.exit(1)


def sha256(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path):

    if not path.exists():

        fail(
            f"No existe:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def read_csv(path: Path):

    if not path.exists():

        fail(
            f"No existe:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows,
):

    if not rows:

        fail(
            f"No hay datos para escribir:\n{path}"
        )

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
        float(value)
        for value in values
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


def safe_float(value):

    if value in [
        None,
        "",
    ]:
        return None

    return float(value)


def fmt_metric(
    mean,
    std,
    n_runs,
):

    if int(n_runs) > 1:

        return (
            f"{float(mean):.4f} "
            f"± {float(std):.4f}"
        )

    return f"{float(mean):.4f}"


# =====================================================================
# VERIFY HOLDOUT
# =====================================================================

def verify_holdout():

    freeze = read_json(
        HOLDOUT_MANIFEST
    )


    if (
        str(
            freeze.get(
                "status",
                ""
            )
        ).upper()
        != "FROZEN"
    ):

        fail(
            "El holdout no está marcado "
            "como FROZEN."
        )


    fingerprint = freeze.get(
        "dataset_fingerprint_sha256"
    )


    if fingerprint != EXPECTED_FINGERPRINT:

        fail(
            "Fingerprint incorrecto.\n\n"
            f"Esperado:\n"
            f"{EXPECTED_FINGERPRINT}\n\n"
            f"Actual:\n"
            f"{fingerprint}"
        )


    holdout_rows = read_csv(
        HOLDOUT_CSV
    )


    if len(holdout_rows) != 220:

        fail(
            f"El holdout contiene "
            f"{len(holdout_rows)} registros; "
            "esperaba 220."
        )


    real = sum(
        int(row["label"]) == 0
        for row in holdout_rows
    )

    fake = sum(
        int(row["label"]) == 1
        for row in holdout_rows
    )


    if real != 100 or fake != 120:

        fail(
            "Distribución del holdout incorrecta:\n"
            f"real={real}, fake={fake}"
        )


    print(
        "Holdout fingerprint :",
        fingerprint,
    )

    print(
        "Holdout             : "
        "100 real + 120 fake = 220"
    )

    print(
        "Integridad lógica   : OK"
    )


    return fingerprint


# =====================================================================
# SOURCE FILES
# =====================================================================

def build_source_files():

    paths = [
        HOLDOUT_MANIFEST,
        HOLDOUT_CSV,

        WAVLM_INITIAL_JSON,
        CNN_JSON,

        DOMAIN_SUMMARY_JSON,
        DOMAIN_GENERATORS_CSV,

        ROBUST_MODES_CSV,
        ROBUST_GENERATORS_CSV,
    ]


    # Domain adapt per-seed generator results
    for seed in SEEDS:

        paths.append(
            DOMAIN_ROOT
            / f"seed_{seed}"
            / "by_generator.csv"
        )

        paths.append(
            DOMAIN_ROOT
            / f"seed_{seed}"
            / "metrics.json"
        )


    # Robust suite per seed
    for mode in [
        "control",
        "robust",
        "gate_only",
        "channel_mix",
    ]:

        for seed in SEEDS:

            paths.append(
                ROBUST_ROOT
                / mode
                / f"seed_{seed}"
                / "by_generator.csv"
            )

            paths.append(
                ROBUST_ROOT
                / mode
                / f"seed_{seed}"
                / "metrics.json"
            )


    source_rows = []


    for path in paths:

        if not path.exists():

            fail(
                "Falta un resultado necesario:\n"
                f"{path}"
            )


        source_rows.append({

            "relative_path":
                path
                .relative_to(ROOT)
                .as_posix(),

            "sha256":
                sha256(path),

            "bytes":
                path.stat().st_size,
        })


    return source_rows


# =====================================================================
# SINGLE-RUN DETECTORS
# =====================================================================

def parse_single_detector(
    detector_id,
    json_path,
):

    data = read_json(
        json_path
    )


    # ---------------------------------------------------------
    # Verify embedded fingerprint
    # ---------------------------------------------------------

    if detector_id == "wavlm_initial":

        embedded = (
            data
            .get("dataset", {})
            .get(
                "fingerprint_sha256"
            )
        )

    else:

        embedded = data.get(
            "dataset_fingerprint"
        )


    if embedded != EXPECTED_FINGERPRINT:

        fail(
            f"{detector_id}: "
            "fingerprint embebido incorrecto.\n"
            f"{embedded}"
        )


    metrics = data[
        "global_metrics"
    ]


    generator_data = data[
        "by_generator"
    ]


    fake_values = [

        float(
            row["primary_value"]
        )

        for row
        in generator_data

        if row["generator"]
        in FAKE_GENERATORS
    ]


    macro_fake = statistics.mean(
        fake_values
    )


    result = {

        "detector_id":
            detector_id,

        "display_name":
            DISPLAY_NAMES[
                detector_id
            ],

        "family":
            (
                "CNN"
                if detector_id
                == "cnn_baseline"
                else "WavLM"
            ),

        "n_runs":
            1,

        "seeds":
            "",

        "accuracy_mean":
            metrics["accuracy"],

        "accuracy_std":
            0.0,

        "balanced_accuracy_mean":
            metrics[
                "balanced_accuracy"
            ],

        "balanced_accuracy_std":
            0.0,

        "precision_mean":
            metrics["precision"],

        "precision_std":
            0.0,

        "recall_fake_mean":
            metrics["recall"],

        "recall_fake_std":
            0.0,

        "specificity_real_mean":
            metrics["specificity"],

        "specificity_real_std":
            0.0,

        "f1_mean":
            metrics["f1"],

        "f1_std":
            0.0,

        "roc_auc_mean":
            metrics["roc_auc"],

        "roc_auc_std":
            0.0,

        "eer_mean":
            metrics["eer"],

        "eer_std":
            0.0,

        "macro_fake_recall_mean":
            macro_fake,

        "macro_fake_recall_std":
            0.0,

        "threshold_mean":
            data["threshold"],

        "threshold_std":
            0.0,
    }


    generator_rows = []


    for row in generator_data:

        generator_rows.append({

            "detector_id":
                detector_id,

            "display_name":
                DISPLAY_NAMES[
                    detector_id
                ],

            "generator":
                row["generator"],

            "metric":
                row[
                    "primary_metric"
                ],

            "mean":
                float(
                    row[
                        "primary_value"
                    ]
                ),

            "std":
                0.0,

            "n_runs":
                1,
        })


    return (
        result,
        generator_rows,
    )


# =====================================================================
# DOMAIN ADAPT
# =====================================================================

def parse_domain_adapt():

    summary = read_json(
        DOMAIN_SUMMARY_JSON
    )


    if (
        summary.get(
            "dataset_fingerprint"
        )
        != EXPECTED_FINGERPRINT
    ):

        fail(
            "Domain-adapt tiene "
            "fingerprint incorrecto."
        )


    metrics = summary[
        "metrics"
    ]


    # ---------------------------------------------------------
    # Calculate exact macro fake recall by seed
    # ---------------------------------------------------------

    macro_per_seed = []


    for seed in SEEDS:

        rows = read_csv(

            DOMAIN_ROOT
            / f"seed_{seed}"
            / "by_generator.csv"

        )


        values = [

            float(
                row["primary_value"]
            )

            for row in rows

            if row["generator"]
            in FAKE_GENERATORS
        ]


        if len(values) != 6:

            fail(
                f"Domain adapt seed {seed}: "
                "no encuentro 6 generadores fake."
            )


        macro_per_seed.append(
            statistics.mean(
                values
            )
        )


    macro_mean, macro_std = (
        mean_std(
            macro_per_seed
        )
    )


    result = {

        "detector_id":
            "wavlm_domain_adapt_real",

        "display_name":
            DISPLAY_NAMES[
                "wavlm_domain_adapt_real"
            ],

        "family":
            "WavLM",

        "n_runs":
            3,

        "seeds":
            "42;123;2026",

        "accuracy_mean":
            metrics[
                "accuracy"
            ]["mean"],

        "accuracy_std":
            metrics[
                "accuracy"
            ]["std"],

        "balanced_accuracy_mean":
            metrics[
                "balanced_accuracy"
            ]["mean"],

        "balanced_accuracy_std":
            metrics[
                "balanced_accuracy"
            ]["std"],

        "precision_mean":
            metrics[
                "precision"
            ]["mean"],

        "precision_std":
            metrics[
                "precision"
            ]["std"],

        "recall_fake_mean":
            metrics[
                "recall"
            ]["mean"],

        "recall_fake_std":
            metrics[
                "recall"
            ]["std"],

        "specificity_real_mean":
            metrics[
                "specificity"
            ]["mean"],

        "specificity_real_std":
            metrics[
                "specificity"
            ]["std"],

        "f1_mean":
            metrics[
                "f1"
            ]["mean"],

        "f1_std":
            metrics[
                "f1"
            ]["std"],

        "roc_auc_mean":
            metrics[
                "roc_auc"
            ]["mean"],

        "roc_auc_std":
            metrics[
                "roc_auc"
            ]["std"],

        "eer_mean":
            metrics[
                "eer"
            ]["mean"],

        "eer_std":
            metrics[
                "eer"
            ]["std"],

        "macro_fake_recall_mean":
            macro_mean,

        "macro_fake_recall_std":
            macro_std,

        "threshold_mean":
            metrics[
                "threshold"
            ]["mean"],

        "threshold_std":
            metrics[
                "threshold"
            ]["std"],
    }


    generator_summary = read_csv(
        DOMAIN_GENERATORS_CSV
    )


    generator_rows = []


    for row in generator_summary:

        generator_rows.append({

            "detector_id":
                "wavlm_domain_adapt_real",

            "display_name":
                DISPLAY_NAMES[
                    "wavlm_domain_adapt_real"
                ],

            "generator":
                row["generator"],

            "metric":
                row["metric"],

            "mean":
                float(row["mean"]),

            "std":
                float(row["std"]),

            "n_runs":
                3,
        })


    return (
        result,
        generator_rows,
    )


# =====================================================================
# ROBUST SUITE
# =====================================================================

def parse_robust_suite():

    mode_summary = read_csv(
        ROBUST_MODES_CSV
    )

    generator_summary = read_csv(
        ROBUST_GENERATORS_CSV
    )


    mode_to_detector = {

        "control":
            "wavlm_control",

        "robust":
            "wavlm_robust",

        "gate_only":
            "wavlm_gate_only",

        "channel_mix":
            "wavlm_channel_mix",
    }


    detector_rows = []
    generator_rows = []


    for mode, detector_id in (
        mode_to_detector.items()
    ):

        matching = [

            row

            for row
            in mode_summary

            if row["mode"] == mode
        ]


        if len(matching) != 1:

            fail(
                f"No encuentro resumen único "
                f"para mode={mode}"
            )


        row = matching[0]


        # -----------------------------------------------------
        # Exact macro fake recall by seed
        # -----------------------------------------------------

        macro_per_seed = []


        for seed in SEEDS:

            seed_rows = read_csv(

                ROBUST_ROOT
                / mode
                / f"seed_{seed}"
                / "by_generator.csv"

            )


            values = [

                float(item["value"])

                for item
                in seed_rows

                if item["generator"]
                in FAKE_GENERATORS
            ]


            if len(values) != 6:

                fail(
                    f"{mode} seed={seed}: "
                    "no encuentro 6 generadores fake."
                )


            macro_per_seed.append(
                statistics.mean(
                    values
                )
            )


        macro_mean, macro_std = (
            mean_std(
                macro_per_seed
            )
        )


        detector_rows.append({

            "detector_id":
                detector_id,

            "display_name":
                DISPLAY_NAMES[
                    detector_id
                ],

            "family":
                "WavLM",

            "n_runs":
                3,

            "seeds":
                "42;123;2026",

            "accuracy_mean":
                float(
                    row[
                        "accuracy_mean"
                    ]
                ),

            "accuracy_std":
                float(
                    row[
                        "accuracy_std"
                    ]
                ),

            "balanced_accuracy_mean":
                float(
                    row[
                        "balanced_accuracy_mean"
                    ]
                ),

            "balanced_accuracy_std":
                float(
                    row[
                        "balanced_accuracy_std"
                    ]
                ),

            "precision_mean":
                float(
                    row[
                        "precision_mean"
                    ]
                ),

            "precision_std":
                float(
                    row[
                        "precision_std"
                    ]
                ),

            "recall_fake_mean":
                float(
                    row[
                        "recall_mean"
                    ]
                ),

            "recall_fake_std":
                float(
                    row[
                        "recall_std"
                    ]
                ),

            "specificity_real_mean":
                float(
                    row[
                        "specificity_mean"
                    ]
                ),

            "specificity_real_std":
                float(
                    row[
                        "specificity_std"
                    ]
                ),

            "f1_mean":
                float(
                    row[
                        "f1_mean"
                    ]
                ),

            "f1_std":
                float(
                    row[
                        "f1_std"
                    ]
                ),

            "roc_auc_mean":
                float(
                    row[
                        "roc_auc_mean"
                    ]
                ),

            "roc_auc_std":
                float(
                    row[
                        "roc_auc_std"
                    ]
                ),

            "eer_mean":
                float(
                    row[
                        "eer_mean"
                    ]
                ),

            "eer_std":
                float(
                    row[
                        "eer_std"
                    ]
                ),

            "macro_fake_recall_mean":
                macro_mean,

            "macro_fake_recall_std":
                macro_std,

            "threshold_mean":
                float(
                    row[
                        "threshold_mean"
                    ]
                ),

            "threshold_std":
                float(
                    row[
                        "threshold_std"
                    ]
                ),
        })


        matching_generators = [

            item

            for item
            in generator_summary

            if item["mode"] == mode
        ]


        if len(
            matching_generators
        ) != 7:

            fail(
                f"{mode}: esperaba "
                "7 entradas por generador."
            )


        for item in matching_generators:

            generator_rows.append({

                "detector_id":
                    detector_id,

                "display_name":
                    DISPLAY_NAMES[
                        detector_id
                    ],

                "generator":
                    item[
                        "generator"
                    ],

                "metric":
                    item[
                        "metric"
                    ],

                "mean":
                    float(
                        item["mean"]
                    ),

                "std":
                    float(
                        item["std"]
                    ),

                "n_runs":
                    3,
            })


    return (
        detector_rows,
        generator_rows,
    )


# =====================================================================
# VALIDATE GLOBAL RESULTS
# =====================================================================

def validate_results(
    detectors,
    generators,
):

    ids = [
        row["detector_id"]
        for row in detectors
    ]


    if set(ids) != set(
        DETECTOR_ORDER
    ):

        fail(
            "No están exactamente "
            "los 7 detectores esperados.\n\n"
            f"Encontrados:\n{ids}"
        )


    if len(ids) != len(
        set(ids)
    ):

        fail(
            "Hay detectores duplicados."
        )


    # Every detector must have all 7
    # generator/domain rows.
    for detector_id in DETECTOR_ORDER:

        items = [

            row

            for row in generators

            if row[
                "detector_id"
            ] == detector_id
        ]


        names = {
            row["generator"]
            for row in items
        }


        if names != set(
            ALL_GENERATORS
        ):

            fail(
                f"{detector_id}: "
                "faltan generadores.\n"
                f"{names}"
            )


    # Numeric sanity checks
    for row in detectors:

        for key in [

            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_fake_mean",
            "specificity_real_mean",
            "f1_mean",
            "roc_auc_mean",
            "eer_mean",
            "macro_fake_recall_mean",

        ]:

            value = float(
                row[key]
            )


            if not (
                0.0
                <= value
                <= 1.0
            ):

                fail(
                    f"{row['detector_id']} "
                    f"{key} fuera de rango: "
                    f"{value}"
                )


# =====================================================================
# RANKINGS
#
# DESCRIPTIVE ONLY.
#
# IMPORTANT:
# These rankings are NOT model selection for further tuning.
# =====================================================================

def build_rankings(
    detectors,
):

    def descending(
        metric
    ):

        return [

            {
                "rank":
                    index,

                "detector_id":
                    row[
                        "detector_id"
                    ],

                "display_name":
                    row[
                        "display_name"
                    ],

                "value":
                    float(
                        row[metric]
                    ),
            }

            for index, row
            in enumerate(

                sorted(

                    detectors,

                    key=lambda x:
                        float(
                            x[metric]
                        ),

                    reverse=True,
                ),

                start=1,
            )
        ]


    def ascending(
        metric
    ):

        return [

            {
                "rank":
                    index,

                "detector_id":
                    row[
                        "detector_id"
                    ],

                "display_name":
                    row[
                        "display_name"
                    ],

                "value":
                    float(
                        row[metric]
                    ),
            }

            for index, row
            in enumerate(

                sorted(

                    detectors,

                    key=lambda x:
                        float(
                            x[metric]
                        ),
                ),

                start=1,
            )
        ]


    return {

        "roc_auc":
            descending(
                "roc_auc_mean"
            ),

        "balanced_accuracy":
            descending(
                "balanced_accuracy_mean"
            ),

        "f1":
            descending(
                "f1_mean"
            ),

        "specificity_real":
            descending(
                "specificity_real_mean"
            ),

        "recall_fake":
            descending(
                "recall_fake_mean"
            ),

        "macro_fake_recall":
            descending(
                "macro_fake_recall_mean"
            ),

        "eer":
            ascending(
                "eer_mean"
            ),
    }


# =====================================================================
# MARKDOWN TABLES
# =====================================================================

def build_markdown(
    detectors,
    generators,
    fingerprint,
):

    by_id = {
        row["detector_id"]:
            row
        for row in detectors
    }


    lines = []

    lines.append(
        "# Resultados finales de detección"
    )

    lines.append("")

    lines.append(
        f"Holdout externo: **220 audios "
        f"(100 reales + 120 falsos)**."
    )

    lines.append("")

    lines.append(
        f"Fingerprint SHA-256: "
        f"`{fingerprint}`"
    )

    lines.append("")

    lines.append(
        "> Los modelos con tres semillas se "
        "> expresan como media ± desviación estándar. "
        "> CNN y WavLM inicial son ejecuciones únicas."
    )

    lines.append("")

    lines.append(
        "## Tabla global"
    )

    lines.append("")

    lines.append(
        "| Detector | N | Accuracy | Balanced Acc. | "
        "Recall fake | Specificity real | F1 | "
        "ROC-AUC | EER | Macro recall fake |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )


    for detector_id in DETECTOR_ORDER:

        row = by_id[
            detector_id
        ]

        n = int(
            row["n_runs"]
        )


        lines.append(

            "| "
            + row[
                "display_name"
            ]

            + " | "
            + str(n)

            + " | "
            + fmt_metric(
                row[
                    "accuracy_mean"
                ],
                row[
                    "accuracy_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "balanced_accuracy_mean"
                ],
                row[
                    "balanced_accuracy_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "recall_fake_mean"
                ],
                row[
                    "recall_fake_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "specificity_real_mean"
                ],
                row[
                    "specificity_real_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "f1_mean"
                ],
                row[
                    "f1_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "roc_auc_mean"
                ],
                row[
                    "roc_auc_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "eer_mean"
                ],
                row[
                    "eer_std"
                ],
                n,
            )

            + " | "
            + fmt_metric(
                row[
                    "macro_fake_recall_mean"
                ],
                row[
                    "macro_fake_recall_std"
                ],
                n,
            )

            + " |"
        )


    lines.append("")
    lines.append(
        "## Resultados por generador"
    )
    lines.append("")

    header = (
        "| Origen | Métrica | "
        + " | ".join(
            DISPLAY_NAMES[
                detector_id
            ]
            for detector_id
            in DETECTOR_ORDER
        )
        + " |"
    )

    lines.append(
        header
    )

    lines.append(
        "|---|---|"
        + "---:|" * len(
            DETECTOR_ORDER
        )
    )


    generator_map = {

        (
            row["detector_id"],
            row["generator"],
        ):
            row

        for row in generators
    }


    for generator in ALL_GENERATORS:

        metric = (
            "Specificity"
            if generator
            == "VoxPopuli"
            else "Fake recall"
        )


        values = []


        for detector_id in DETECTOR_ORDER:

            row = generator_map[
                (
                    detector_id,
                    generator,
                )
            ]


            values.append(

                fmt_metric(
                    row["mean"],
                    row["std"],
                    row["n_runs"],
                )
            )


        lines.append(

            "| "
            + generator
            + " | "
            + metric
            + " | "
            + " | ".join(
                values
            )
            + " |"
        )


    lines.append("")

    lines.append(
        "## Nota metodológica"
    )

    lines.append("")

    lines.append(
        "Los umbrales de decisión fueron establecidos "
        "antes de la evaluación del holdout externo. "
        "No se recalibró ningún detector utilizando "
        "los 220 audios del conjunto final."
    )

    lines.append("")

    lines.append(
        "Las clasificaciones relativas que puedan "
        "derivarse de estas tablas son descriptivas "
        "y no deben utilizarse para entrenar, "
        "recalibrar o seleccionar nuevas variantes "
        "empleando este mismo holdout."
    )

    lines.append("")


    return "\n".join(
        lines
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Sobrescribe únicamente el resumen "
            "derivado. Nunca modifica el holdout "
            "ni las predicciones originales."
        ),
    )

    args = parser.parse_args()


    # ---------------------------------------------------------
    # Output safety
    # ---------------------------------------------------------

    final_files = [
        DETECTORS_CSV,
        GENERATORS_CSV,
        SOURCES_CSV,
        REPORT_JSON,
        TABLES_MD,
        FREEZE_JSON,
    ]


    existing = [
        path
        for path in final_files
        if path.exists()
    ]


    if existing and not args.force:

        fail(
            "Ya existen resultados finales:\n\n"
            + "\n".join(
                str(path)
                for path
                in existing
            )
            + "\n\nUsa --force únicamente "
            "si quieres reconstruir este resumen "
            "a partir de los mismos resultados."
        )


    print()
    print("=" * 80)
    print(
        "CONSTRUYENDO RESUMEN FINAL "
        "DE DETECCIÓN"
    )
    print("=" * 80)


    fingerprint = verify_holdout()


    # =========================================================
    # SOURCES
    # =========================================================

    source_rows = build_source_files()


    # =========================================================
    # CNN
    # =========================================================

    (
        cnn,
        cnn_generators,

    ) = parse_single_detector(

        "cnn_baseline",

        CNN_JSON,
    )


    # =========================================================
    # INITIAL WAVLM
    # =========================================================

    (
        initial,
        initial_generators,

    ) = parse_single_detector(

        "wavlm_initial",

        WAVLM_INITIAL_JSON,
    )


    # =========================================================
    # ROBUST SUITE
    # =========================================================

    (
        robust_detectors,
        robust_generators,

    ) = parse_robust_suite()


    # =========================================================
    # DOMAIN ADAPT
    # =========================================================

    (
        domain,
        domain_generators,

    ) = parse_domain_adapt()


    # =========================================================
    # MERGE
    # =========================================================

    detectors = [
        cnn,
        initial,
        *robust_detectors,
        domain,
    ]


    detectors = sorted(

        detectors,

        key=lambda row:
            DETECTOR_ORDER.index(
                row["detector_id"]
            )
    )


    generators = [
        *cnn_generators,
        *initial_generators,
        *robust_generators,
        *domain_generators,
    ]


    generators = sorted(

        generators,

        key=lambda row: (

            DETECTOR_ORDER.index(
                row[
                    "detector_id"
                ]
            ),

            ALL_GENERATORS.index(
                row[
                    "generator"
                ]
            ),
        )
    )


    validate_results(
        detectors,
        generators,
    )


    rankings = build_rankings(
        detectors
    )


    # =========================================================
    # WRITE OUTPUTS
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_csv(
        DETECTORS_CSV,
        detectors,
    )


    write_csv(
        GENERATORS_CSV,
        generators,
    )


    write_csv(
        SOURCES_CSV,
        source_rows,
    )


    markdown = build_markdown(

        detectors,

        generators,

        fingerprint,
    )


    with TABLES_MD.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            markdown
        )


    # =========================================================
    # RESULT FINGERPRINT
    # =========================================================

    detector_csv_hash = sha256(
        DETECTORS_CSV
    )

    generator_csv_hash = sha256(
        GENERATORS_CSV
    )

    sources_csv_hash = sha256(
        SOURCES_CSV
    )


    fingerprint_payload = (

        EXPECTED_FINGERPRINT
        + "\n"
        + detector_csv_hash
        + "\n"
        + generator_csv_hash
        + "\n"
        + sources_csv_hash

    )


    results_fingerprint = (
        hashlib.sha256(
            fingerprint_payload.encode(
                "utf-8"
            )
        ).hexdigest()
    )


    # =========================================================
    # MAIN REPORT
    # =========================================================

    report = {

        "name":
            "Final External Detection Results",

        "status":
            "FINAL",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "holdout": {

            "n_total":
                220,

            "n_real":
                100,

            "n_fake":
                120,

            "dataset_fingerprint_sha256":
                fingerprint,
        },

        "methodological_policy": {

            "external_holdout":
                (
                    "Final evaluation only."
                ),

            "thresholds":
                (
                    "Fixed/calibrated before "
                    "external holdout evaluation."
                ),

            "post_hoc_retuning":
                False,

            "post_hoc_seed_selection":
                False,

            "ranking_usage":
                (
                    "Descriptive reporting only; "
                    "not for further tuning on "
                    "this holdout."
                ),
        },

        "detectors":
            detectors,

        "per_generator":
            generators,

        "descriptive_rankings":
            rankings,

        "source_files":
            source_rows,

        "derived_files": {

            "detector_summary": {
                "path":
                    DETECTORS_CSV
                    .relative_to(ROOT)
                    .as_posix(),

                "sha256":
                    detector_csv_hash,
            },

            "generator_summary": {
                "path":
                    GENERATORS_CSV
                    .relative_to(ROOT)
                    .as_posix(),

                "sha256":
                    generator_csv_hash,
            },

            "source_files": {
                "path":
                    SOURCES_CSV
                    .relative_to(ROOT)
                    .as_posix(),

                "sha256":
                    sources_csv_hash,
            },

            "markdown_tables": {
                "path":
                    TABLES_MD
                    .relative_to(ROOT)
                    .as_posix(),

                "sha256":
                    sha256(
                        TABLES_MD
                    ),
            },
        },

        "results_fingerprint_sha256":
            results_fingerprint,
    }


    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # =========================================================
    # FINAL MANIFEST
    # =========================================================

    final_manifest = {

        "status":
            "FROZEN_RESULTS",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "holdout_fingerprint_sha256":
            fingerprint,

        "results_fingerprint_sha256":
            results_fingerprint,

        "files": [

            {
                "file":
                    path.name,

                "relative_path":
                    path
                    .relative_to(ROOT)
                    .as_posix(),

                "sha256":
                    sha256(path),

                "bytes":
                    path.stat().st_size,
            }

            for path in [

                DETECTORS_CSV,
                GENERATORS_CSV,
                SOURCES_CSV,
                REPORT_JSON,
                TABLES_MD,
            ]
        ],

        "rule":
            (
                "These files summarize the final "
                "external evaluation. Do not alter "
                "historical predictions or retune "
                "models using the external holdout."
            ),
    }


    with FREEZE_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # =========================================================
    # TERMINAL REPORT
    # =========================================================

    print()
    print("=" * 80)
    print(
        "RESULTADOS FINALES DE DETECCIÓN"
    )
    print("=" * 80)


    for row in detectors:

        print()

        print(
            row[
                "display_name"
            ]
        )

        print(
            "  Balanced Acc : "
            + fmt_metric(

                row[
                    "balanced_accuracy_mean"
                ],

                row[
                    "balanced_accuracy_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )

        print(
            "  Recall fake  : "
            + fmt_metric(

                row[
                    "recall_fake_mean"
                ],

                row[
                    "recall_fake_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )

        print(
            "  Specificity  : "
            + fmt_metric(

                row[
                    "specificity_real_mean"
                ],

                row[
                    "specificity_real_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )

        print(
            "  F1           : "
            + fmt_metric(

                row[
                    "f1_mean"
                ],

                row[
                    "f1_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )

        print(
            "  ROC-AUC      : "
            + fmt_metric(

                row[
                    "roc_auc_mean"
                ],

                row[
                    "roc_auc_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )

        print(
            "  EER          : "
            + fmt_metric(

                row[
                    "eer_mean"
                ],

                row[
                    "eer_std"
                ],

                row[
                    "n_runs"
                ],
            )
        )


    print()
    print("=" * 80)

    print(
        "MEJORES RESULTADOS DESCRIPTIVOS"
    )

    print("=" * 80)


    best_auc = rankings[
        "roc_auc"
    ][0]

    best_bal = rankings[
        "balanced_accuracy"
    ][0]

    best_spec = rankings[
        "specificity_real"
    ][0]

    best_recall = rankings[
        "recall_fake"
    ][0]

    best_eer = rankings[
        "eer"
    ][0]


    print(
        f"ROC-AUC          : "
        f"{best_auc['display_name']} "
        f"({best_auc['value']:.4f})"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_bal['display_name']} "
        f"({best_bal['value']:.4f})"
    )

    print(
        f"Specificity real : "
        f"{best_spec['display_name']} "
        f"({best_spec['value']:.4f})"
    )

    print(
        f"Recall fake      : "
        f"{best_recall['display_name']} "
        f"({best_recall['value']:.4f})"
    )

    print(
        f"Menor EER        : "
        f"{best_eer['display_name']} "
        f"({best_eer['value']:.4f})"
    )


    print()
    print(
        "Fingerprint resultados:"
    )

    print(
        results_fingerprint
    )


    print()
    print(
        f"Tabla detectores : {DETECTORS_CSV}"
    )

    print(
        f"Por generador    : {GENERATORS_CSV}"
    )

    print(
        f"Tablas TFG       : {TABLES_MD}"
    )

    print(
        f"Informe JSON     : {REPORT_JSON}"
    )

    print(
        f"Manifest final   : {FREEZE_JSON}"
    )


    print()
    print(
        "EVALUACIÓN EXTERNA FORMALMENTE CERRADA."
    )

    print(
        "A partir de aquí no se reajustan "
        "modelos ni thresholds usando este holdout."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()