from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import soundfile as sf


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
BUILD_DIR = ROOT / "_external_fake_build"

OUTPUT_DIR = BUILD_DIR / "voxcpm2_outputs"
GENERATION_CSV = BUILD_DIR / "voxcpm2_generation.csv"
GENERATION_MANIFEST = BUILD_DIR / "voxcpm2_generation_manifest.json"
JOBS_CSV = BUILD_DIR / "voxcpm2_jobs.csv"

FINAL_ROOT = ROOT / "external_final_fake"
FINAL_DIR = FINAL_ROOT / "fake" / "voxcpm2"

METADATA_CSV = FINAL_ROOT / "metadata_voxcpm2.csv"
FREEZE_MANIFEST = FINAL_ROOT / "freeze_manifest_voxcpm2.json"


# =====================================================================
# EXPECTED
# =====================================================================

EXPECTED_COUNT = 20
EXPECTED_SPEAKERS = 10
EXPECTED_SR = 48000

EXPECTED_GENERATOR = "VoxCPM2"
EXPECTED_MODEL = "openbmb/VoxCPM2"

EXPECTED_CFG = 2.0
EXPECTED_STEPS = 10
EXPECTED_NORMALIZE = False
EXPECTED_DENOISE = False
EXPECTED_RETRY_BADCASE = False


# =====================================================================
# UTILS
# =====================================================================

def fail(msg):
    print()
    print("ERROR:", msg)
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


def inspect_audio(path: Path):

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    if audio.size == 0:
        raise ValueError("audio vacío")

    if not np.isfinite(audio).all():
        raise ValueError("audio contiene NaN o Inf")

    frames = audio.shape[0]
    channels = audio.shape[1]
    duration = frames / sr

    abs_audio = np.abs(audio)

    peak = float(
        np.max(abs_audio)
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    clipped_fraction = float(
        np.mean(
            abs_audio >= 0.999
        )
    )

    return {
        "sample_rate": int(sr),
        "channels": int(channels),
        "frames": int(frames),
        "duration": float(duration),
        "peak": peak,
        "rms": rms,
        "clipped_fraction": clipped_fraction,
    }


def csv_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


# =====================================================================
# PRECHECK
# =====================================================================

for path in [
    OUTPUT_DIR,
    GENERATION_CSV,
    GENERATION_MANIFEST,
    JOBS_CSV,
]:
    if not path.exists():
        fail(
            f"No existe:\n{path}"
        )


for path in [
    FINAL_DIR,
    METADATA_CSV,
    FREEZE_MANIFEST,
]:
    if path.exists():
        fail(
            f"Ya existe:\n{path}\n"
            "No se sobrescribe."
        )


# =====================================================================
# GENERATION MANIFEST
# =====================================================================

with GENERATION_MANIFEST.open(
    "r",
    encoding="utf-8"
) as f:

    gen_manifest = json.load(f)


if (
    gen_manifest.get("generator")
    != EXPECTED_GENERATOR
):
    fail(
        "Generator inesperado: "
        f"{gen_manifest.get('generator')}"
    )


if (
    int(gen_manifest.get("count", -1))
    != EXPECTED_COUNT
):
    fail(
        "Manifest indica "
        f"{gen_manifest.get('count')} audios."
    )


if (
    int(gen_manifest.get("sample_rate", -1))
    != EXPECTED_SR
):
    fail(
        "Sample rate inesperado en manifest: "
        f"{gen_manifest.get('sample_rate')}"
    )


if (
    gen_manifest.get(
        "reference_transcript_used"
    )
    is not False
):
    fail(
        "El manifest no confirma "
        "reference_transcript_used=False."
    )


if (
    gen_manifest.get("clone_mode")
    != "reference_only_voice_cloning"
):
    fail(
        "clone_mode inesperado: "
        f"{gen_manifest.get('clone_mode')}"
    )


# =====================================================================
# GENERATION PARAMETERS
# =====================================================================

params = (
    gen_manifest.get(
        "generation_parameters"
    )
    or {}
)


if float(
    params.get("cfg_value", -1)
) != EXPECTED_CFG:
    fail(
        "cfg_value inesperado."
    )


if int(
    params.get(
        "inference_timesteps",
        -1
    )
) != EXPECTED_STEPS:
    fail(
        "inference_timesteps inesperado."
    )


if (
    params.get("normalize")
    is not EXPECTED_NORMALIZE
):
    fail(
        "normalize inesperado."
    )


if (
    params.get("denoise")
    is not EXPECTED_DENOISE
):
    fail(
        "denoise inesperado."
    )


if (
    params.get("retry_badcase")
    is not EXPECTED_RETRY_BADCASE
):
    fail(
        "retry_badcase inesperado."
    )


# =====================================================================
# MODEL IDENTITY
# =====================================================================

model_identity = (
    gen_manifest.get(
        "model_identity"
    )
    or {}
)


if (
    model_identity.get(
        "requested_model_id"
    )
    != EXPECTED_MODEL
):
    fail(
        "Modelo inesperado: "
        f"{model_identity.get('requested_model_id')}"
    )


# =====================================================================
# GENERATION CSV
# =====================================================================

with GENERATION_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != EXPECTED_COUNT:
    fail(
        f"voxcpm2_generation.csv tiene "
        f"{len(rows)} filas."
    )


speakers = sorted({
    row["speaker"]
    for row in rows
})


if len(speakers) != EXPECTED_SPEAKERS:
    fail(
        f"Esperaba {EXPECTED_SPEAKERS} speakers, "
        f"hay {len(speakers)}."
    )


for speaker in speakers:

    n = sum(
        row["speaker"] == speaker
        for row in rows
    )

    if n != 2:
        fail(
            f"{speaker}: {n} audios; "
            "esperaba 2."
        )


# =====================================================================
# VALIDATE CSV SETTINGS
# =====================================================================

for row in rows:

    if (
        row["generator"].strip()
        != EXPECTED_GENERATOR
    ):
        fail(
            f"{row['id']}: generator incorrecto."
        )


    if (
        row["model"].strip()
        != EXPECTED_MODEL
    ):
        fail(
            f"{row['id']}: model incorrecto."
        )


    if (
        row["clone_mode"].strip()
        != "reference_only_voice_cloning"
    ):
        fail(
            f"{row['id']}: clone_mode incorrecto."
        )


    if (
        float(row["cfg_value"])
        != EXPECTED_CFG
    ):
        fail(
            f"{row['id']}: cfg_value incorrecto."
        )


    if (
        int(row["inference_timesteps"])
        != EXPECTED_STEPS
    ):
        fail(
            f"{row['id']}: steps incorrectos."
        )


    if (
        csv_bool(row["normalize"])
        != EXPECTED_NORMALIZE
    ):
        fail(
            f"{row['id']}: normalize incorrecto."
        )


    if (
        csv_bool(row["denoise"])
        != EXPECTED_DENOISE
    ):
        fail(
            f"{row['id']}: denoise incorrecto."
        )


    if (
        csv_bool(row["retry_badcase"])
        != EXPECTED_RETRY_BADCASE
    ):
        fail(
            f"{row['id']}: retry_badcase incorrecto."
        )


# =====================================================================
# OUTPUT FILES
# =====================================================================

outputs = sorted(
    OUTPUT_DIR.glob("*.wav")
)


if len(outputs) != EXPECTED_COUNT:
    fail(
        f"Hay {len(outputs)} WAV; "
        f"esperaba {EXPECTED_COUNT}."
    )


output_by_name = {
    path.name: path
    for path in outputs
}


expected_names = {
    row["file"].strip()
    for row in rows
}


actual_names = {
    path.name
    for path in outputs
}


extras = (
    actual_names
    - expected_names
)

missing = (
    expected_names
    - actual_names
)


if extras:
    fail(
        "Outputs extra:\n"
        + "\n".join(
            sorted(extras)
        )
    )


if missing:
    fail(
        "Outputs ausentes:\n"
        + "\n".join(
            sorted(missing)
        )
    )


# =====================================================================
# QC
# =====================================================================

validated = []
warnings = []


for row in rows:

    filename = (
        row["file"].strip()
    )

    path = (
        output_by_name[
            filename
        ]
    )


    # -------------------------------------------------------------
    # HASH
    # -------------------------------------------------------------

    current_hash = sha256(
        path
    )

    expected_hash = (
        row["sha256"].strip()
    )


    if current_hash != expected_hash:
        fail(
            "El WAV cambió después de generarse:\n"
            f"{path}"
        )


    # -------------------------------------------------------------
    # AUDIO
    # -------------------------------------------------------------

    info = inspect_audio(
        path
    )


    if (
        info["sample_rate"]
        != EXPECTED_SR
    ):
        fail(
            f"{filename}: "
            f"{info['sample_rate']} Hz != "
            f"{EXPECTED_SR} Hz"
        )


    if (
        int(row["sample_rate"])
        != info["sample_rate"]
    ):
        fail(
            f"{filename}: "
            "sample_rate del CSV "
            "no coincide con el WAV."
        )


    if info["duration"] < 0.5:
        fail(
            f"{filename}: duración demasiado corta "
            f"({info['duration']:.3f}s)"
        )


    if info["rms"] < 1e-5:
        fail(
            f"{filename}: "
            "audio prácticamente silencioso."
        )


    if info["duration"] > 30:
        warnings.append(
            f"{filename}: duración alta "
            f"({info['duration']:.2f}s)"
        )


    if (
        info["clipped_fraction"]
        > 0.01
    ):
        warnings.append(
            f"{filename}: "
            f"{info['clipped_fraction'] * 100:.2f}% "
            "de muestras cerca de clipping"
        )


    validated.append({
        "row": row,
        "path": path,
        "info": info,
        "sha256": current_hash,
    })


# =====================================================================
# CREATE FINAL
# =====================================================================

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=False
)


metadata_rows = []
manifest_files = []


# =====================================================================
# FREEZE WAVS
# =====================================================================

for i, item in enumerate(
    validated,
    start=1
):

    row = item["row"]
    info = item["info"]

    final_name = (
        f"voxcpm2_{i:03d}.wav"
    )

    destination = (
        FINAL_DIR
        / final_name
    )


    shutil.copy2(
        item["path"],
        destination
    )


    frozen_hash = sha256(
        destination
    )


    if (
        frozen_hash
        != item["sha256"]
    ):
        fail(
            f"Hash distinto tras copiar "
            f"{final_name}"
        )


    # -------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------

    metadata_rows.append({

        "file":
            final_name,

        "label":
            1,

        "class":
            "fake",

        "generator":
            EXPECTED_GENERATOR,

        "model":
            EXPECTED_MODEL,

        "speaker":
            row["speaker"],

        "reference_file":
            row["reference_file"],

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        # Conservado como metadata.
        # NO utilizado para generar.
        "reference_text":
            row["reference_text"],

        "reference_text_used":
            False,

        "target_text":
            row["target_text"],

        "language":
            row["language"],

        "clone_mode":
            row["clone_mode"],

        "seed":
            row["seed"],

        "seed_strategy":
            params.get(
                "seed_strategy",
                "global_python_numpy_torch_rng_before_each_generation"
            ),

        "cfg_value":
            row["cfg_value"],

        "inference_timesteps":
            row[
                "inference_timesteps"
            ],

        "normalize":
            row["normalize"],

        "denoise":
            row["denoise"],

        "retry_badcase":
            row["retry_badcase"],

        "sample_rate":
            info["sample_rate"],

        "channels":
            info["channels"],

        "duration_s":
            f"{info['duration']:.6f}",

        "peak":
            f"{info['peak']:.8f}",

        "rms":
            f"{info['rms']:.8f}",

        "clipped_fraction":
            f"{info['clipped_fraction']:.8f}",

        "sha256":
            frozen_hash,
    })


    manifest_files.append({

        "file":
            final_name,

        "speaker":
            row["speaker"],

        "seed":
            int(row["seed"]),

        "sha256":
            frozen_hash,

        "bytes":
            destination.stat().st_size,

        "duration_s":
            info["duration"],

        "sample_rate":
            info["sample_rate"],
    })


# =====================================================================
# METADATA CSV
# =====================================================================

with METADATA_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            metadata_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        metadata_rows
    )


# =====================================================================
# FREEZE MANIFEST
# =====================================================================

freeze_manifest = {

    "name":
        "External final fake holdout - VoxCPM2",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "generator":
        EXPECTED_GENERATOR,

    "model":
        EXPECTED_MODEL,

    "count":
        len(manifest_files),

    "speakers":
        speakers,

    "audios_per_speaker":
        2,

    "sample_rate":
        EXPECTED_SR,


    # -------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------

    "generation": {

        "clone_mode":
            "reference_only_voice_cloning",

        "reference_transcript_used":
            False,

        "cfg_value":
            EXPECTED_CFG,

        "inference_timesteps":
            EXPECTED_STEPS,

        "normalize":
            EXPECTED_NORMALIZE,

        "denoise":
            EXPECTED_DENOISE,

        "retry_badcase":
            EXPECTED_RETRY_BADCASE,

        "seed_strategy":
            params.get(
                "seed_strategy",
                "global_python_numpy_torch_rng_before_each_generation"
            ),

        "optimize":
            params.get(
                "optimize"
            ),

        "device":
            gen_manifest.get(
                "device"
            ),
    },


    # -------------------------------------------------------------
    # MODEL IDENTITY
    # -------------------------------------------------------------

    "model_identity":
        model_identity,


    # -------------------------------------------------------------
    # SOURCE FILES
    # -------------------------------------------------------------

    "source_files": {

        "jobs_csv": {

            "path_at_generation":
                str(JOBS_CSV),

            "sha256":
                sha256(JOBS_CSV),
        },

        "generation_csv": {

            "path_at_generation":
                str(
                    GENERATION_CSV
                ),

            "sha256":
                sha256(
                    GENERATION_CSV
                ),
        },

        "generation_manifest": {

            "path_at_generation":
                str(
                    GENERATION_MANIFEST
                ),

            "sha256":
                sha256(
                    GENERATION_MANIFEST
                ),
        },
    },


    # -------------------------------------------------------------
    # FINAL FILES
    # -------------------------------------------------------------

    "files":
        manifest_files,
}


with FREEZE_MANIFEST.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 72)
print("VOXCPM2 EXTERNAL FINAL HOLDOUT — CONGELADO")
print("=" * 72)

print(
    f"Generator : {EXPECTED_GENERATOR}"
)

print(
    f"Model     : {EXPECTED_MODEL}"
)

print(
    f"Speakers  : {len(speakers)}"
)

print(
    f"Audios    : {len(manifest_files)}"
)

print(
    f"SR        : {EXPECTED_SR} Hz"
)

print()

print(
    f"WAVs      : {FINAL_DIR}"
)

print(
    f"Metadata  : {METADATA_CSV}"
)

print(
    f"Manifest  : {FREEZE_MANIFEST}"
)

print()


if warnings:

    print("AVISOS DE QC:")

    for warning in warnings:
        print(
            " -",
            warning
        )

else:

    print(
        "QC         : OK"
    )


print()
print(
    "NO regenerar ni sobrescribir estos WAVs."
)

print("=" * 72)