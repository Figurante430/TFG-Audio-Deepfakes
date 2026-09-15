from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WavLMModel


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection"
)

GLOBAL_CSV = (
    ROOT
    / "external_final_holdout"
    / "metadata_external_final.csv"
)

GLOBAL_MANIFEST = (
    ROOT
    / "external_final_holdout"
    / "freeze_manifest_external_final.json"
)

EXPECTED_FINGERPRINT = (
    "3fed1b260a404af75b21aab8a57f1cf"
    "2dd0dbc8967d2695b45780095008f6021"
)

MODEL_NAME = "microsoft/wavlm-base-plus"

TARGET_SR = 16000
CLIP_SECONDS = 4.0
NUM_SAMPLES = int(
    TARGET_SR * CLIP_SECONDS
)

# ============================================================
# MUY IMPORTANTE:
# calibrado ANTES del holdout externo.
# NO se modifica usando estos 220 audios.
# ============================================================

FIXED_THRESHOLD = 0.89501953125


OUTPUT_ROOT = (
    ROOT
    / "external_results"
    / "wavlm_final"
)

PREPROCESS_DIR = (
    OUTPUT_ROOT
    / "preprocessed_16k"
)

PREDICTIONS_CSV = (
    OUTPUT_ROOT
    / "wavlm_predictions.csv"
)

METRICS_JSON = (
    OUTPUT_ROOT
    / "wavlm_metrics.json"
)

BY_GENERATOR_CSV = (
    OUTPUT_ROOT
    / "wavlm_by_generator.csv"
)

PREPROCESS_MANIFEST = (
    OUTPUT_ROOT
    / "preprocessing_manifest.csv"
)

CONFUSION_PNG = (
    OUTPUT_ROOT
    / "wavlm_confusion_matrix.png"
)


EXPECTED_TOTAL = 220
EXPECTED_REAL = 100
EXPECTED_FAKE = 120


# =====================================================================
# UTILS
# =====================================================================

def fail(message):

    print()
    print("=" * 78)
    print("ERROR")
    print("=" * 78)
    print(message)
    print("=" * 78)

    sys.exit(1)


def sha256(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path: Path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(path: Path, rows):

    if not rows:
        raise RuntimeError(
            f"No hay filas para guardar en {path}"
        )

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
# HOLDOUT INTEGRITY
# =====================================================================

def verify_holdout():

    if not GLOBAL_CSV.exists():
        fail(
            f"No existe:\n{GLOBAL_CSV}"
        )

    if not GLOBAL_MANIFEST.exists():
        fail(
            f"No existe:\n{GLOBAL_MANIFEST}"
        )


    with GLOBAL_MANIFEST.open(
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)


    fingerprint = (
        manifest.get(
            "dataset_fingerprint_sha256"
        )
    )


    if fingerprint != EXPECTED_FINGERPRINT:

        fail(
            "FINGERPRINT DEL HOLDOUT DISTINTO.\n\n"
            f"Esperado:\n{EXPECTED_FINGERPRINT}\n\n"
            f"Encontrado:\n{fingerprint}"
        )


    if (
        str(
            manifest.get("status", "")
        ).upper()
        != "FROZEN"
    ):

        fail(
            "El manifest global no está marcado "
            "como FROZEN."
        )


    rows = read_csv(
        GLOBAL_CSV
    )


    if len(rows) != EXPECTED_TOTAL:

        fail(
            f"Metadata global contiene "
            f"{len(rows)} filas; "
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
            f"Reales={real}; "
            f"esperaba {EXPECTED_REAL}."
        )

    if fake != EXPECTED_FAKE:
        fail(
            f"Fakes={fake}; "
            f"esperaba {EXPECTED_FAKE}."
        )


    manifest_files = {

        item["relative_path"]:
            item

        for item in manifest.get(
            "files",
            []
        )
    }


    if len(manifest_files) != EXPECTED_TOTAL:

        fail(
            "El freeze manifest global no "
            "enumera los 220 audios."
        )


    print()
    print("=" * 78)
    print("VERIFICANDO INTEGRIDAD DEL HOLDOUT")
    print("=" * 78)


    for i, row in enumerate(
        rows,
        start=1
    ):

        relative_path = (
            row["relative_path"]
        )

        source = (
            ROOT
            / Path(relative_path)
        )


        if not source.exists():

            fail(
                f"Falta WAV congelado:\n"
                f"{source}"
            )


        expected_hash = (
            row["sha256"]
            .strip()
            .lower()
        )


        manifest_item = (
            manifest_files.get(
                relative_path
            )
        )


        if manifest_item is None:

            fail(
                "El WAV no aparece en "
                "freeze_manifest_external_final.json:\n"
                f"{relative_path}"
            )


        manifest_hash = (
            manifest_item["sha256"]
            .strip()
            .lower()
        )


        if manifest_hash != expected_hash:

            fail(
                "CSV global y manifest global "
                "no coinciden:\n"
                f"{relative_path}"
            )


        current_hash = (
            sha256(source)
            .lower()
        )


        if current_hash != expected_hash:

            fail(
                "WAV DEL HOLDOUT MODIFICADO:\n"
                f"{source}\n\n"
                f"Congelado:\n{expected_hash}\n"
                f"Actual:\n{current_hash}"
            )


        if (
            i % 25 == 0
            or i == EXPECTED_TOTAL
        ):

            print(
                f"\rVerificados: "
                f"{i}/{EXPECTED_TOTAL}",
                end="",
                flush=True
            )


    print()
    print(
        "Fingerprint :",
        fingerprint
    )
    print(
        "Integridad  : OK"
    )


    return rows, manifest


# =====================================================================
# CHECKPOINT DISCOVERY
# =====================================================================

def collect_threshold_values(
    obj,
    path=""
):

    found = []


    if isinstance(obj, dict):

        for key, value in obj.items():

            next_path = (
                f"{path}.{key}"
                if path
                else key
            )


            if (
                "threshold"
                in key.lower()
                and isinstance(
                    value,
                    (int, float)
                )
            ):

                found.append(
                    (
                        next_path,
                        float(value)
                    )
                )


            found.extend(
                collect_threshold_values(
                    value,
                    next_path
                )
            )


    elif isinstance(obj, list):

        for index, value in enumerate(
            obj
        ):

            found.extend(
                collect_threshold_values(
                    value,
                    f"{path}[{index}]"
                )
            )


    return found


def metrics_contains_threshold(
    metrics_path: Path,
    threshold: float
):

    try:

        with metrics_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:

        return False, []


    matches = [

        (key, value)

        for key, value
        in collect_threshold_values(data)

        if abs(
            value - threshold
        ) < 1e-10
    ]


    return bool(matches), matches


def discover_checkpoint():

    experiments = (
        ROOT
        / "experiments"
    )


    if not experiments.exists():

        return []


    candidates = {}


    for metrics_path in experiments.rglob(
        "metrics.json"
    ):

        matches, values = (
            metrics_contains_threshold(
                metrics_path,
                FIXED_THRESHOLD
            )
        )


        if not matches:
            continue


        checkpoint = (
            metrics_path.parent
            / "best_model.pt"
        )


        if not checkpoint.exists():
            continue


        candidates[
            str(
                checkpoint.resolve()
            )
        ] = {

            "checkpoint":
                checkpoint,

            "metrics":
                metrics_path,

            "matches":
                values,
        }


    return list(
        candidates.values()
    )


def resolve_checkpoint(
    explicit_checkpoint
):

    if explicit_checkpoint is not None:

        checkpoint = (
            explicit_checkpoint.resolve()
        )


        if not checkpoint.exists():

            fail(
                f"No existe checkpoint:\n"
                f"{checkpoint}"
            )


        sibling_metrics = (
            checkpoint.parent
            / "metrics.json"
        )


        if sibling_metrics.exists():

            matches, _ = (
                metrics_contains_threshold(
                    sibling_metrics,
                    FIXED_THRESHOLD
                )
            )


            if not matches:

                fail(
                    "El metrics.json asociado al "
                    "checkpoint NO contiene el "
                    "threshold fijo esperado:\n\n"
                    f"Checkpoint:\n"
                    f"{checkpoint}\n\n"
                    f"Threshold esperado:\n"
                    f"{FIXED_THRESHOLD}"
                )


        return (
            checkpoint,
            sibling_metrics
            if sibling_metrics.exists()
            else None
        )


    candidates = (
        discover_checkpoint()
    )


    if len(candidates) == 0:

        fail(
            "No he encontrado automáticamente "
            "un best_model.pt cuyo metrics.json "
            "contenga el threshold "
            f"{FIXED_THRESHOLD}.\n\n"
            "Ejecuta de nuevo indicando:\n"
            "  --checkpoint <ruta_al_best_model.pt>"
        )


    if len(candidates) > 1:

        print()
        print(
            "Se han encontrado varios "
            "checkpoints compatibles:"
        )
        print()

        for i, item in enumerate(
            candidates,
            start=1
        ):

            print(
                f"[{i}] {item['checkpoint']}"
            )

            print(
                f"    metrics: "
                f"{item['metrics']}"
            )

            print(
                f"    coincidencias: "
                f"{item['matches']}"
            )


        fail(
            "Hay más de un candidato.\n"
            "No elegiré uno automáticamente.\n\n"
            "Ejecuta el script con:\n"
            "  --checkpoint <ruta>"
        )


    item = candidates[0]

    return (
        item["checkpoint"],
        item["metrics"]
    )


# =====================================================================
# MODEL
# =====================================================================

class WavLMClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()


        try:

            self.wavlm = (
                WavLMModel.from_pretrained(
                    MODEL_NAME,
                    local_files_only=True
                )
            )

        except Exception as exc:

            raise RuntimeError(
                "No puedo cargar "
                f"{MODEL_NAME} desde la cache local.\n"
                "Usa el mismo entorno con el que "
                "entrenaste WavLM.\n\n"
                f"{type(exc).__name__}: {exc}"
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
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                1
            ),
        )


    def forward(
        self,
        input_values,
        attention_mask
    ):

        outputs = self.wavlm(

            input_values=
                input_values,

            attention_mask=
                attention_mask,
        )


        hidden = (
            outputs
            .last_hidden_state
        )


        feature_mask = (
            self.wavlm
            ._get_feature_vector_attention_mask(
                hidden.shape[1],
                attention_mask
            )
            .to(
                hidden.device
            )
        )


        mask = (
            feature_mask
            .unsqueeze(-1)
            .to(
                hidden.dtype
            )
        )


        pooled = (
            hidden * mask
        ).sum(
            dim=1
        )


        denom = (
            mask.sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        pooled = (
            pooled
            / denom
        )


        return (
            self.head(
                pooled
            )
            .squeeze(-1)
        )


def get_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        nn.Module
    ):

        return (
            checkpoint
            .state_dict()
        )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Formato de checkpoint "
            "no reconocido."
        )


    for key in [

        "model_state_dict",
        "state_dict",
        "model_state",
        "model",

    ]:

        value = checkpoint.get(
            key
        )


        if isinstance(
            value,
            dict
        ):

            return value


        if isinstance(
            value,
            nn.Module
        ):

            return (
                value
                .state_dict()
            )


    if (
        checkpoint
        and all(
            isinstance(key, str)
            for key in checkpoint.keys()
        )
        and any(
            torch.is_tensor(value)
            for value
            in checkpoint.values()
        )
    ):

        return checkpoint


    raise RuntimeError(
        "No encuentro state_dict "
        "en el checkpoint."
    )


def normalize_state_dict_keys(
    state
):

    result = {}


    for key, value in state.items():

        new_key = key


        for prefix in [
            "module.",
            "model.",
        ]:

            if new_key.startswith(
                prefix
            ):

                new_key = (
                    new_key[
                        len(prefix):
                    ]
                )


        if new_key.startswith(
            "backbone."
        ):

            new_key = (
                "wavlm."
                + new_key[
                    len(
                        "backbone."
                    ):
                ]
            )


        if new_key.startswith(
            "classifier."
        ):

            new_key = (
                "head."
                + new_key[
                    len(
                        "classifier."
                    ):
                ]
            )


        result[
            new_key
        ] = value


    return result


def load_model(
    checkpoint_path,
    device
):

    print()
    print("=" * 78)
    print("CARGANDO WAVLM")
    print("=" * 78)

    print(
        f"Checkpoint:\n"
        f"  {checkpoint_path}"
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )


    model = WavLMClassifier()


    state = (
        normalize_state_dict_keys(
            get_state_dict(
                checkpoint
            )
        )
    )


    try:

        model.load_state_dict(
            state,
            strict=True
        )

    except RuntimeError as exc:

        fail(
            "El checkpoint no coincide "
            "exactamente con la arquitectura "
            "WavLM esperada.\n\n"
            f"{exc}"
        )


    model.to(
        device
    )

    model.eval()


    metadata = {}


    if isinstance(
        checkpoint,
        dict
    ):

        for key in [

            "epoch",
            "stage",
            "held_out",
            "val_auc",
            "seed",

        ]:

            if key in checkpoint:

                value = checkpoint[
                    key
                ]

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool
                    )
                ):

                    metadata[
                        key
                    ] = value


    print(
        f"Checkpoint SHA256 : "
        f"{sha256(checkpoint_path)}"
    )

    print(
        f"Device            : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )


    print(
        f"Threshold FIJO    : "
        f"{FIXED_THRESHOLD:.12f}"
    )


    return (
        model,
        checkpoint,
        metadata
    )


# =====================================================================
# PREPROCESS
# =====================================================================

def valid_normalized_wav(
    path
):

    if not path.exists():
        return False


    try:

        info = sf.info(
            str(path)
        )

    except Exception:

        return False


    return (
        info.samplerate
        == TARGET_SR

        and info.channels
        == 1

        and info.subtype
        == "PCM_16"

        and info.frames
        > 0
    )


def normalize_with_ffmpeg(
    source,
    destination,
    ffmpeg
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    cmd = [

        ffmpeg,

        "-hide_banner",

        "-loglevel",
        "error",

        "-y",

        "-i",
        str(source),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(TARGET_SR),

        "-c:a",
        "pcm_s16le",

        str(destination),
    ]


    completed = subprocess.run(

        cmd,

        stdout=
            subprocess.PIPE,

        stderr=
            subprocess.PIPE,

        text=True,
    )


    if completed.returncode != 0:

        fail(
            "FFmpeg falló:\n"
            f"{source}\n\n"
            f"{completed.stderr}"
        )


    if not valid_normalized_wav(
        destination
    ):

        fail(
            "El WAV normalizado no cumple "
            "16 kHz / mono / PCM16:\n"
            f"{destination}"
        )


def prepare_audio_cache(
    rows
):

    ffmpeg = shutil.which(
        "ffmpeg"
    )


    if ffmpeg is None:

        fail(
            "No encuentro ffmpeg en PATH."
        )


    PREPROCESS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    prepared_rows = []
    preprocess_rows = []


    print()
    print("=" * 78)
    print("PREPROCESADO EXTERNO → 16 kHz MONO PCM16")
    print("=" * 78)


    for index, row in enumerate(
        rows,
        start=1
    ):

        source = (
            ROOT
            / Path(
                row[
                    "relative_path"
                ]
            )
        )


        source_hash = (
            row[
                "sha256"
            ]
            .strip()
            .lower()
        )


        cache_name = (

            f"{int(row['id']):03d}_"
            f"{source_hash[:12]}_"
            f"{source.stem}.wav"

        )


        destination = (
            PREPROCESS_DIR
            / cache_name
        )


        if not destination.exists():

            normalize_with_ffmpeg(

                source,

                destination,

                ffmpeg
            )


        elif not valid_normalized_wav(
            destination
        ):

            fail(
                "Existe un fichero de cache "
                "pero es inválido:\n"
                f"{destination}\n\n"
                "Bórralo y vuelve a ejecutar."
            )


        derived_hash = (
            sha256(
                destination
            )
        )


        enriched = dict(
            row
        )

        enriched[
            "_normalized_path"
        ] = str(
            destination
        )

        enriched[
            "_normalized_sha256"
        ] = derived_hash


        prepared_rows.append(
            enriched
        )


        preprocess_rows.append({

            "id":
                row["id"],

            "source_relative_path":
                row[
                    "relative_path"
                ],

            "source_sha256":
                source_hash,

            "normalized_path":
                str(
                    destination
                ),

            "normalized_sha256":
                derived_hash,

            "target_sample_rate":
                TARGET_SR,

            "target_channels":
                1,

            "target_subtype":
                "PCM_16",
        })


        if (
            index % 20 == 0
            or index == len(rows)
        ):

            print(
                f"\rPreparados: "
                f"{index}/{len(rows)}",
                end="",
                flush=True
            )


    print()


    write_csv(
        PREPROCESS_MANIFEST,
        preprocess_rows
    )


    return prepared_rows


# =====================================================================
# EXACT TEST-TIME CROP / PAD
# =====================================================================

def load_for_wavlm(
    path
):

    audio, sr = sf.read(

        str(path),

        dtype="float32",

        always_2d=True
    )


    if sr != TARGET_SR:

        raise RuntimeError(
            f"{path}: SR={sr}, "
            f"esperaba {TARGET_SR}."
        )


    # [samples, channels] -> mono
    audio = (
        audio
        .mean(axis=1)
    )


    audio = np.asarray(
        audio,
        dtype=np.float32
    )


    n = len(
        audio
    )


    # ============================================================
    # MISMA LÓGICA DEL DATASET DE ENTRENAMIENTO:
    #
    # - >= 4 s -> crop central
    # - < 4 s  -> padding simétrico
    # ============================================================

    if n >= NUM_SAMPLES:

        start = (
            n - NUM_SAMPLES
        ) // 2


        waveform = (

            audio[
                start:
                start + NUM_SAMPLES
            ]

        )


        attention_mask = (
            np.ones(
                NUM_SAMPLES,
                dtype=np.int64
            )
        )


    else:

        missing = (
            NUM_SAMPLES
            - n
        )


        left = (
            missing
            // 2
        )


        right = (
            missing
            - left
        )


        waveform = np.pad(

            audio,

            (
                left,
                right
            ),

            mode="constant"
        )


        attention_mask = (
            np.zeros(
                NUM_SAMPLES,
                dtype=np.int64
            )
        )


        attention_mask[
            left:
            left + n
        ] = 1


    return (

        waveform.astype(
            np.float32
        ),

        attention_mask
    )


# =====================================================================
# INFERENCE
# =====================================================================

@torch.inference_mode()
def predict(
    model,
    rows,
    device,
    batch_size,
    checkpoint_hash
):

    predictions = []


    print()
    print("=" * 78)
    print("INFERENCIA WAVLM — HOLDOUT EXTERNO")
    print("=" * 78)


    total = len(
        rows
    )


    for start in range(
        0,
        total,
        batch_size
    ):

        batch_rows = (
            rows[
                start:
                start + batch_size
            ]
        )


        waveforms = []
        masks = []


        for row in batch_rows:

            waveform, mask = (
                load_for_wavlm(

                    Path(
                        row[
                            "_normalized_path"
                        ]
                    )

                )
            )


            waveforms.append(
                waveform
            )

            masks.append(
                mask
            )


        x = torch.tensor(

            np.stack(
                waveforms
            ),

            dtype=
                torch.float32,

            device=
                device
        )


        attention_mask = (
            torch.tensor(

                np.stack(
                    masks
                ),

                dtype=
                    torch.long,

                device=
                    device
            )
        )


        if device.type == "cuda":

            with torch.autocast(

                device_type="cuda",

                dtype=
                    torch.float16,

                enabled=True
            ):

                logits = model(
                    x,
                    attention_mask
                )

        else:

            logits = model(
                x,
                attention_mask
            )


        probs = (

            torch.sigmoid(
                logits.float()
            )
            .cpu()
            .numpy()

        )


        for row, probability in zip(
            batch_rows,
            probs
        ):

            probability = float(
                probability
            )


            label = int(
                row["label"]
            )


            prediction = (

                1

                if probability
                >= FIXED_THRESHOLD

                else 0
            )


            if (
                label == 0
                and prediction == 1
            ):

                error_type = "FP"

            elif (
                label == 1
                and prediction == 0
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
                    row[
                        "file"
                    ],

                "label":
                    label,

                "class":
                    row[
                        "class"
                    ],

                "generator":
                    row[
                        "generator"
                    ],

                "speaker":
                    row[
                        "speaker"
                    ],

                "source_sample_rate":
                    row[
                        "sample_rate"
                    ],

                "source_sha256":
                    row[
                        "sha256"
                    ],

                "normalized_sha256":
                    row[
                        "_normalized_sha256"
                    ],

                "prob_fake":
                    probability,

                "threshold":
                    FIXED_THRESHOLD,

                "pred_label":
                    prediction,

                "pred_class":
                    (
                        "fake"
                        if prediction == 1
                        else "real"
                    ),

                "correct":
                    int(
                        prediction == label
                    ),

                "error_type":
                    error_type,

                "checkpoint_sha256":
                    checkpoint_hash,
            })


        done = min(
            start + batch_size,
            total
        )


        print(
            f"\rProcesados: "
            f"{done}/{total}",
            end="",
            flush=True
        )


    print()


    return predictions


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
    ).astype(
        int
    )


    tp = int(
        np.sum(
            (preds == 1)
            & (labels == 1)
        )
    )


    tn = int(
        np.sum(
            (preds == 0)
            & (labels == 0)
        )
    )


    fp = int(
        np.sum(
            (preds == 1)
            & (labels == 0)
        )
    )


    fn = int(
        np.sum(
            (preds == 0)
            & (labels == 1)
        )
    )


    total = len(
        labels
    )


    accuracy = (

        (tp + tn) / total

        if total
        else 0.0
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

        2
        * precision
        * recall
        / (
            precision
            + recall
        )

        if (
            precision
            + recall
        )
        else 0.0
    )


    balanced_accuracy = (
        (
            recall
            + specificity
        )
        / 2.0
    )


    fpr = (

        fp / (fp + tn)

        if (fp + tn)
        else 0.0
    )


    fnr = (

        fn / (fn + tp)

        if (fn + tp)
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


def roc_auc_pairwise(
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


    positives = probs[
        labels == 1
    ]

    negatives = probs[
        labels == 0
    ]


    if (
        len(positives) == 0
        or len(negatives) == 0
    ):

        return float(
            "nan"
        )


    greater = 0.0
    total = 0


    for positive in positives:

        greater += float(
            np.sum(
                positive
                > negatives
            )
        )

        greater += (
            0.5
            * float(
                np.sum(
                    positive
                    == negatives
                )
            )
        )


        total += len(
            negatives
        )


    return (
        greater
        / total
    )


def eer_metric(
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


    thresholds = np.unique(

        np.concatenate(
            [
                np.array(
                    [
                        -1e-9,
                        1.0 + 1e-9
                    ]
                ),
                probs,
            ]
        )
    )


    best = None


    for threshold in thresholds:

        metrics = confusion_metrics(

            labels,

            probs,

            float(
                threshold
            )
        )


        fpr = (
            metrics["fpr"]
        )

        fnr = (
            metrics["fnr"]
        )


        candidate = (

            abs(
                fpr - fnr
            ),

            (
                fpr + fnr
            )
            / 2.0,

            float(
                threshold
            ),
        )


        if (
            best is None
            or candidate[0]
            < best[0]
        ):

            best = candidate


    return {

        "eer":
            float(
                best[1]
            ),

        "eer_threshold":
            float(
                best[2]
            ),
    }


# =====================================================================
# BY GENERATOR
# =====================================================================

def build_generator_metrics(
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


    output = []


    order = [

        "VoxPopuli",
        "RVC",
        "Qwen3-TTS",
        "Fun-CosyVoice3",
        "Confucius4-TTS",
        "OpenVoice V2",
        "VoxCPM2",
    ]


    for generator in order:

        rows = groups[
            generator
        ]


        probs = np.asarray(

            [
                float(
                    row["prob_fake"]
                )

                for row in rows
            ],

            dtype=float
        )


        labels = [

            int(
                row["label"]
            )

            for row in rows
        ]


        predictions_array = [

            int(
                row["pred_label"]
            )

            for row in rows
        ]


        correct = sum(

            pred == label

            for pred, label
            in zip(
                predictions_array,
                labels
            )
        )


        label_value = (
            labels[0]
        )


        if label_value == 0:

            tn = sum(
                pred == 0
                for pred
                in predictions_array
            )

            fp = (
                len(rows)
                - tn
            )

            detection_metric = (
                tn / len(rows)
            )

            detection_name = (
                "specificity"
            )

            errors = fp


        else:

            tp = sum(
                pred == 1
                for pred
                in predictions_array
            )

            fn = (
                len(rows)
                - tp
            )

            detection_metric = (
                tp / len(rows)
            )

            detection_name = (
                "fake_recall"
            )

            errors = fn


        output.append({

            "generator":
                generator,

            "true_label":
                label_value,

            "n":
                len(rows),

            "correct":
                correct,

            "errors":
                errors,

            "accuracy":
                correct
                / len(rows),

            "primary_metric":
                detection_name,

            "primary_value":
                detection_metric,

            "prob_fake_mean":
                float(
                    np.mean(
                        probs
                    )
                ),

            "prob_fake_median":
                float(
                    np.median(
                        probs
                    )
                ),

            "prob_fake_min":
                float(
                    np.min(
                        probs
                    )
                ),

            "prob_fake_max":
                float(
                    np.max(
                        probs
                    )
                ),
        })


    return output


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

def save_confusion_matrix(
    metrics
):

    try:

        import matplotlib.pyplot as plt

    except Exception as exc:

        print(
            "AVISO: no se genera matriz "
            "de confusión PNG porque "
            "matplotlib no está disponible:"
        )

        print(
            exc
        )

        return


    matrix = np.array(
        [
            [
                metrics["tn"],
                metrics["fp"]
            ],
            [
                metrics["fn"],
                metrics["tp"]
            ],
        ]
    )


    fig, ax = plt.subplots(
        figsize=(5, 4)
    )


    image = ax.imshow(
        matrix
    )


    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "Pred REAL",
            "Pred FAKE"
        ]
    )

    ax.set_yticklabels(
        [
            "REAL",
            "FAKE"
        ]
    )


    for i in range(2):

        for j in range(2):

            ax.text(

                j,
                i,

                str(
                    matrix[i, j]
                ),

                ha="center",
                va="center"
            )


    ax.set_title(
        "WavLM — External Final Holdout"
    )


    fig.colorbar(
        image,
        ax=ax
    )


    fig.tight_layout()


    fig.savefig(
        CONFUSION_PNG,
        dpi=160
    )


    plt.close(
        fig
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Evaluación FINAL de WavLM sobre "
            "el holdout externo congelado."
        )
    )


    parser.add_argument(

        "--checkpoint",

        type=Path,

        default=None,

        help=(
            "best_model.pt. Si se omite, "
            "se intenta localizar automáticamente "
            "el checkpoint cuyo metrics.json "
            "contenga el threshold fijo."
        )
    )


    parser.add_argument(

        "--batch-size",

        type=int,

        default=4
    )


    parser.add_argument(

        "--force",

        action="store_true",

        help=(
            "Permite sobrescribir únicamente "
            "los RESULTADOS de evaluación. "
            "Nunca modifica el holdout."
        )
    )


    args = parser.parse_args()


    # ============================================================
    # SAFETY
    # ============================================================

    result_files = [

        PREDICTIONS_CSV,
        METRICS_JSON,
        BY_GENERATOR_CSV,
    ]


    existing = [

        path

        for path in result_files

        if path.exists()
    ]


    if existing and not args.force:

        fail(
            "Ya existen resultados WavLM:\n"
            + "\n".join(
                str(path)
                for path
                in existing
            )
            + "\n\nNo se sobrescriben."
        )


    # ============================================================
    # VERIFY DATASET
    # ============================================================

    rows, global_manifest = (
        verify_holdout()
    )


    # ============================================================
    # CHECKPOINT
    # ============================================================

    checkpoint_path, metrics_path = (
        resolve_checkpoint(
            args.checkpoint
        )
    )


    checkpoint_hash = (
        sha256(
            checkpoint_path
        )
    )


    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    # ============================================================
    # MODEL
    # ============================================================

    (
        model,
        checkpoint_raw,
        checkpoint_metadata

    ) = load_model(

        checkpoint_path,

        device
    )


    # ============================================================
    # PREPROCESS
    # ============================================================

    prepared_rows = (
        prepare_audio_cache(
            rows
        )
    )


    # ============================================================
    # INFERENCE
    # ============================================================

    predictions = predict(

        model,

        prepared_rows,

        device,

        args.batch_size,

        checkpoint_hash
    )


    # ============================================================
    # METRICS
    # ============================================================

    labels = np.asarray(

        [
            int(
                row["label"]
            )

            for row
            in predictions
        ],

        dtype=int
    )


    probs = np.asarray(

        [
            float(
                row["prob_fake"]
            )

            for row
            in predictions
        ],

        dtype=float
    )


    global_metrics = (
        confusion_metrics(

            labels,

            probs,

            FIXED_THRESHOLD
        )
    )


    global_metrics[
        "roc_auc"
    ] = roc_auc_pairwise(

        labels,

        probs
    )


    global_metrics.update(

        eer_metric(
            labels,
            probs
        )
    )


    global_metrics[
        "prob_fake_real_mean"
    ] = float(

        np.mean(
            probs[
                labels == 0
            ]
        )
    )


    global_metrics[
        "prob_fake_fake_mean"
    ] = float(

        np.mean(
            probs[
                labels == 1
            ]
        )
    )


    # ============================================================
    # GENERATORS
    # ============================================================

    generator_rows = (
        build_generator_metrics(
            predictions
        )
    )


    # ============================================================
    # SAVE
    # ============================================================

    OUTPUT_ROOT.mkdir(
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


    report = {

        "evaluation":
            "external_final_holdout",

        "model":
            MODEL_NAME,

        "threshold":
            FIXED_THRESHOLD,

        "threshold_policy":
            (
                "Fixed before external final "
                "holdout evaluation. "
                "Not calibrated on these 220 audios."
            ),

        "dataset": {

            "n_total":
                len(predictions),

            "n_real":
                int(
                    np.sum(
                        labels == 0
                    )
                ),

            "n_fake":
                int(
                    np.sum(
                        labels == 1
                    )
                ),

            "fingerprint_sha256":
                EXPECTED_FINGERPRINT,
        },

        "checkpoint": {

            "path":
                str(
                    checkpoint_path
                ),

            "sha256":
                checkpoint_hash,

            "associated_metrics_json":
                (
                    str(
                        metrics_path
                    )
                    if metrics_path
                    else None
                ),

            "metadata":
                checkpoint_metadata,
        },

        "preprocessing": {

            "normalization":
                (
                    "FFmpeg mono, 16000 Hz, PCM16 "
                    "on derived copies only"
                ),

            "clip_seconds":
                CLIP_SECONDS,

            "long_audio":
                "central crop",

            "short_audio":
                "symmetric zero padding",

            "attention_mask":
                True,
        },

        "global_metrics":
            global_metrics,

        "by_generator":
            generator_rows,
    }


    with METRICS_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )


    save_confusion_matrix(
        global_metrics
    )


    # ============================================================
    # TERMINAL REPORT
    # ============================================================

    print()
    print("=" * 78)
    print("WAVLM — EXTERNAL FINAL HOLDOUT")
    print("=" * 78)

    print(
        f"Audios          : "
        f"{len(predictions)}"
    )

    print(
        f"Threshold FIJO  : "
        f"{FIXED_THRESHOLD:.12f}"
    )

    print(
        f"Accuracy        : "
        f"{global_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc.   : "
        f"{global_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision       : "
        f"{global_metrics['precision']:.4f}"
    )

    print(
        f"Recall fake     : "
        f"{global_metrics['recall']:.4f}"
    )

    print(
        f"Specificity     : "
        f"{global_metrics['specificity']:.4f}"
    )

    print(
        f"F1              : "
        f"{global_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{global_metrics['roc_auc']:.4f}"
    )

    print(
        f"EER             : "
        f"{global_metrics['eer']:.4f}"
    )

    print()

    print(
        f"TN={global_metrics['tn']}  "
        f"FP={global_metrics['fp']}  "
        f"FN={global_metrics['fn']}  "
        f"TP={global_metrics['tp']}"
    )

    print()
    print(
        "RESULTADO POR ORIGEN"
    )
    print("-" * 78)


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

    if CONFUSION_PNG.exists():

        print(
            f"Matriz conf. : "
            f"{CONFUSION_PNG}"
        )


    print()
    print(
        "HOLDOUT NO MODIFICADO."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()