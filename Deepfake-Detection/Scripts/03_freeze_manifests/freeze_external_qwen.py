from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import soundfile as sf


ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

BUILD_DIR = ROOT / "_external_fake_build"

OUTPUT_DIR = BUILD_DIR / "qwen_outputs"
GENERATION_CSV = BUILD_DIR / "qwen_generation.csv"
GENERATION_MANIFEST = BUILD_DIR / "qwen_generation_manifest.json"
JOBS_CSV = BUILD_DIR / "qwen_jobs.csv"

FINAL_ROOT = ROOT / "external_final_fake"
FINAL_DIR = FINAL_ROOT / "fake" / "qwen"

METADATA_CSV = FINAL_ROOT / "metadata_qwen.csv"
FREEZE_MANIFEST = FINAL_ROOT / "freeze_manifest_qwen.json"

EXPECTED_COUNT = 20
EXPECTED_SPEAKERS = 10
EXPECTED_SR = 24000

EXPECTED_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

EXPECTED_SNAPSHOT = (
    "fd4b254389122332181a7c3db7f27e918eec64e3"
)


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
        always_2d=True,
        dtype="float32"
    )

    if audio.size == 0:
        raise ValueError("audio vacío")

    if not np.isfinite(audio).all():
        raise ValueError("contiene NaN o Inf")

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
        fail(f"No existe: {path}")


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
# MANIFEST DE GENERACIÓN
# =====================================================================

with GENERATION_MANIFEST.open(
    "r",
    encoding="utf-8"
) as f:
    gen_manifest = json.load(f)


if gen_manifest.get("model_id") != EXPECTED_MODEL:
    fail(
        "Modelo inesperado:\n"
        f"{gen_manifest.get('model_id')}"
    )


if (
    gen_manifest.get("model_snapshot")
    != EXPECTED_SNAPSHOT
):
    fail(
        "Snapshot inesperado:\n"
        f"{gen_manifest.get('model_snapshot')}"
    )


if gen_manifest.get("count") != EXPECTED_COUNT:
    fail(
        f"Manifest indica "
        f"{gen_manifest.get('count')} audios."
    )


# =====================================================================
# CSV GENERACIÓN
# =====================================================================

with GENERATION_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    generated_rows = list(
        csv.DictReader(f)
    )


if len(generated_rows) != EXPECTED_COUNT:
    fail(
        f"qwen_generation.csv tiene "
        f"{len(generated_rows)} filas."
    )


speakers = sorted({
    row["speaker"]
    for row in generated_rows
})


if len(speakers) != EXPECTED_SPEAKERS:
    fail(
        f"Esperaba {EXPECTED_SPEAKERS} speakers, "
        f"hay {len(speakers)}."
    )


for speaker in speakers:

    n = sum(
        row["speaker"] == speaker
        for row in generated_rows
    )

    if n != 2:
        fail(
            f"{speaker}: {n} audios; "
            "esperaba 2."
        )


# =====================================================================
# OUTPUTS
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
    x.name: x
    for x in outputs
}


# =====================================================================
# QC
# =====================================================================

validated = []
warnings = []


for row in generated_rows:

    filename = row["file"].strip()

    if filename not in output_by_name:
        fail(
            f"No encuentro output: {filename}"
        )

    path = output_by_name[filename]

    current_hash = sha256(path)

    recorded_hash = row[
        "sha256"
    ].strip()


    if current_hash != recorded_hash:
        fail(
            "El WAV cambió después de generarse:\n"
            f"{path}"
        )


    try:
        info = inspect_audio(path)

    except Exception as e:
        fail(
            f"Audio inválido:\n{path}\n{e}"
        )


    if info["sample_rate"] != EXPECTED_SR:
        fail(
            f"{filename}: "
            f"{info['sample_rate']} Hz != "
            f"{EXPECTED_SR} Hz"
        )


    if info["duration"] < 0.5:
        fail(
            f"{filename}: duración demasiado corta "
            f"({info['duration']:.3f}s)"
        )


    if info["rms"] < 1e-5:
        fail(
            f"{filename}: audio prácticamente silencioso"
        )


    if info["duration"] > 30:
        warnings.append(
            f"{filename}: duración alta "
            f"({info['duration']:.2f}s)"
        )


    if info["clipped_fraction"] > 0.01:
        warnings.append(
            f"{filename}: "
            f"{info['clipped_fraction']*100:.2f}% "
            "de muestras cerca de clipping"
        )


    # Comprobar coherencia con el CSV de generación
    csv_sr = int(row["sample_rate"])

    if csv_sr != info["sample_rate"]:
        fail(
            f"{filename}: SR del CSV no coincide."
        )


    validated.append({
        "row": row,
        "path": path,
        "info": info,
        "sha256": current_hash,
    })


# =====================================================================
# NOMBRES DUPLICADOS / EXTRAS
# =====================================================================

expected_names = {
    row["file"]
    for row in generated_rows
}


actual_names = {
    x.name
    for x in outputs
}


extras = actual_names - expected_names
missing = expected_names - actual_names


if extras:
    fail(
        "Outputs extra:\n"
        + "\n".join(sorted(extras))
    )


if missing:
    fail(
        "Outputs ausentes:\n"
        + "\n".join(sorted(missing))
    )


# =====================================================================
# CONGELAR
# =====================================================================

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=False
)


metadata_rows = []
manifest_files = []


for i, item in enumerate(
    validated,
    start=1
):

    row = item["row"]
    info = item["info"]

    final_name = f"qwen_{i:03d}.wav"

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


    if frozen_hash != item["sha256"]:
        fail(
            f"Hash distinto tras copiar "
            f"{final_name}"
        )


    metadata_rows.append({

        "file": final_name,

        "label": 1,

        "class": "fake",

        "generator": "Qwen3-TTS",

        "model":
            EXPECTED_MODEL,

        "model_snapshot":
            EXPECTED_SNAPSHOT,

        "speaker":
            row["speaker"],

        "reference_file":
            row["reference_file"],

        "original_reference_file":
            row["original_reference_file"],

        "reference_text":
            row["reference_text"],

        "target_text":
            row["target_text"],

        "language":
            row["language"],

        "clone_mode":
            row["clone_mode"],

        "seed":
            row["seed"],

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
# METADATA
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
        "External final fake holdout - Qwen3-TTS",

    "status":
        "FROZEN",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "generator":
        "Qwen3-TTS",

    "count":
        len(manifest_files),

    "speakers":
        speakers,

    "audios_per_speaker":
        2,

    "model": {

        "id":
            EXPECTED_MODEL,

        "snapshot":
            EXPECTED_SNAPSHOT,

        "dtype":
            gen_manifest.get(
                "dtype"
            ),

        "attention":
            gen_manifest.get(
                "attention"
            ),
    },

    "generation": {

        "language":
            gen_manifest.get(
                "language"
            ),

        "clone_mode":
            gen_manifest.get(
                "clone_mode"
            ),

        "x_vector_only_mode":
            gen_manifest.get(
                "x_vector_only_mode"
            ),

        "non_streaming_mode":
            gen_manifest.get(
                "non_streaming_mode"
            ),

        "parameters":
            gen_manifest.get(
                "generation_parameters"
            ),
    },

    "source_files": {

        "jobs_csv": {
            "path_at_generation":
                str(JOBS_CSV),

            "sha256":
                sha256(JOBS_CSV),
        },

        "generation_csv": {
            "path_at_generation":
                str(GENERATION_CSV),

            "sha256":
                sha256(GENERATION_CSV),
        },

        "generation_manifest": {
            "path_at_generation":
                str(GENERATION_MANIFEST),

            "sha256":
                sha256(
                    GENERATION_MANIFEST
                ),
        },
    },

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
print("QWEN EXTERNAL FINAL HOLDOUT — CONGELADO")
print("=" * 72)

print(f"Generator : Qwen3-TTS")
print(f"Model     : {EXPECTED_MODEL}")
print(f"Snapshot  : {EXPECTED_SNAPSHOT}")
print(f"Speakers  : {len(speakers)}")
print(f"Audios    : {len(manifest_files)}")
print(f"SR        : {EXPECTED_SR} Hz")

print()
print(f"WAVs      : {FINAL_DIR}")
print(f"Metadata  : {METADATA_CSV}")
print(f"Manifest  : {FREEZE_MANIFEST}")

print()

if warnings:

    print("AVISOS DE QC:")

    for warning in warnings:
        print(" -", warning)

else:
    print("QC         : OK")


print()
print("NO regenerar ni sobrescribir estos WAVs.")
print("=" * 72)