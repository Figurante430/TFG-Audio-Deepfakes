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
# CONFIGURACIÓN
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

BUILD_DIR = ROOT / "_external_fake_build"
SOURCE_DIR = BUILD_DIR / "rvc_sources"
OUTPUT_DIR = BUILD_DIR / "rvc_outputs"
SELECTION_CSV = BUILD_DIR / "rvc_selection.csv"

FINAL_ROOT = ROOT / "external_final_fake"
FINAL_DIR = FINAL_ROOT / "fake" / "rvc"
METADATA_CSV = FINAL_ROOT / "metadata_rvc.csv"
MANIFEST_JSON = FINAL_ROOT / "freeze_manifest_rvc.json"

RVC_ROOT = Path(
    r"C:\Users\gonza\TFG\Retrieval-based-Voice-Conversion-WebUI"
)

MODEL = RVC_ROOT / "assets" / "weights" / "Mario_40.pth"

INDEX = (
    RVC_ROOT
    / "logs"
    / "Mario_40"
    / "added_IVF2528_Flat_nprobe_1_Mario_40_v2.index"
)

EXPECTED_COUNT = 20
EXPECTED_SR = 40000

PARAMETERS = {
    "generator": "RVC",
    "model": "Mario_40",
    "model_info": "200epoch",
    "rvc_version": "v2",
    "checkpoint_sample_rate": "40k",
    "checkpoint_f0": 0,
    "speaker_id": 0,
    "pitch": 0,
    "f0_method_cli_default": "rmvpe",
    "index_rate": 0.75,
    "rms_mix_rate": 1.0,
    "protect": 0.33,
    "resample_sr": 0,
    "output_format": "wav",
}


# =====================================================================
# UTILIDADES
# =====================================================================

def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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

    peak = float(np.max(abs_audio))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    clipped_fraction = float(
        np.mean(abs_audio >= 0.999)
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


def fail(message):
    print()
    print("ERROR:", message)
    sys.exit(1)


# =====================================================================
# COMPROBACIONES PREVIAS
# =====================================================================

for path in [
    SOURCE_DIR,
    OUTPUT_DIR,
    SELECTION_CSV,
    MODEL,
    INDEX,
]:
    if not path.exists():
        fail(f"No existe: {path}")


# No sobrescribimos nunca un holdout congelado.
if FINAL_DIR.exists():
    fail(
        f"Ya existe el directorio final:\n{FINAL_DIR}\n"
        "No se sobrescribe."
    )

if METADATA_CSV.exists():
    fail(
        f"Ya existe:\n{METADATA_CSV}\n"
        "No se sobrescribe."
    )

if MANIFEST_JSON.exists():
    fail(
        f"Ya existe:\n{MANIFEST_JSON}\n"
        "No se sobrescribe."
    )


# =====================================================================
# LEER SELECCIÓN ORIGINAL
# =====================================================================

with SELECTION_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    selection = list(csv.DictReader(f))


if len(selection) != EXPECTED_COUNT:
    fail(
        f"rvc_selection.csv contiene {len(selection)} filas; "
        f"esperaba {EXPECTED_COUNT}."
    )


# =====================================================================
# COMPROBAR OUTPUTS
# =====================================================================

outputs = sorted(OUTPUT_DIR.glob("*.wav"))

if len(outputs) != EXPECTED_COUNT:
    fail(
        f"Hay {len(outputs)} WAV en rvc_outputs; "
        f"esperaba {EXPECTED_COUNT}."
    )

output_by_stem = {
    x.stem: x
    for x in outputs
}


# =====================================================================
# VALIDACIÓN INDIVIDUAL
# =====================================================================

validated = []
warnings = []

for n, row in enumerate(selection, start=1):

    input_name = row["rvc_input"]

    source = SOURCE_DIR / input_name

    if not source.exists():
        fail(f"Falta source: {source}")

    # La CLI batch de RVC mantiene el stem.
    expected_output_stem = Path(input_name).stem

    if expected_output_stem not in output_by_stem:
        fail(
            f"No encuentro output para:\n"
            f"{input_name}\n"
            f"Stem esperado: {expected_output_stem}"
        )

    output = output_by_stem[expected_output_stem]

    # ---------------------------------------------------------------
    # Comprobar que la fuente no cambió
    # ---------------------------------------------------------------

    source_hash = sha256(source)

    recorded_hash = row.get("staged_sha256", "").strip()

    if recorded_hash and source_hash != recorded_hash:
        fail(
            f"La fuente cambió desde la selección:\n"
            f"{source}"
        )

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    try:
        src_info = inspect_audio(source)
        out_info = inspect_audio(output)

    except Exception as e:
        fail(
            f"Audio inválido:\n"
            f"{output}\n"
            f"{e}"
        )

    if out_info["sample_rate"] != EXPECTED_SR:
        fail(
            f"{output.name}: sample rate "
            f"{out_info['sample_rate']} != {EXPECTED_SR}"
        )

    if out_info["duration"] <= 0.25:
        fail(
            f"{output.name}: duración demasiado corta "
            f"({out_info['duration']:.3f}s)"
        )

    if out_info["rms"] < 1e-5:
        fail(
            f"{output.name}: parece silencioso "
            f"(RMS={out_info['rms']})"
        )

    duration_ratio = (
        out_info["duration"]
        / src_info["duration"]
    )

    if not 0.80 <= duration_ratio <= 1.20:
        warnings.append(
            f"{output.name}: duración RVC/source = "
            f"{duration_ratio:.3f}"
        )

    if out_info["clipped_fraction"] > 0.01:
        warnings.append(
            f"{output.name}: "
            f"{out_info['clipped_fraction'] * 100:.2f}% "
            f"de muestras cerca de clipping"
        )

    validated.append({
        "number": n,
        "selection": row,
        "source": source,
        "output": output,
        "source_hash": source_hash,
        "output_hash": sha256(output),
        "source_info": src_info,
        "output_info": out_info,
        "duration_ratio": duration_ratio,
    })


# =====================================================================
# COMPROBAR OUTPUTS EXTRAÑOS
# =====================================================================

expected_stems = {
    Path(row["rvc_input"]).stem
    for row in selection
}

actual_stems = set(output_by_stem)

extras = actual_stems - expected_stems

if extras:
    fail(
        "Hay outputs que no pertenecen a la selección:\n"
        + "\n".join(sorted(extras))
    )


# =====================================================================
# CONGELAR
# =====================================================================

FINAL_DIR.mkdir(parents=True, exist_ok=False)

metadata_rows = []
manifest_files = []


for item in validated:

    n = item["number"]
    row = item["selection"]

    destination_name = f"rvc_{n:03d}.wav"
    destination = FINAL_DIR / destination_name

    shutil.copy2(
        item["output"],
        destination
    )

    frozen_hash = sha256(destination)

    if frozen_hash != item["output_hash"]:
        fail(
            f"Hash distinto después de copiar "
            f"{destination_name}"
        )

    src = item["source_info"]
    out = item["output_info"]

    metadata_rows.append({
        "file": destination_name,
        "label": 1,
        "class": "fake",
        "generator": "RVC",
        "model": "Mario_40",
        "speaker_id": 0,

        "source_file": row["original_file"],
        "source_staged_file": row["rvc_input"],
        "source_speaker": row["speaker"],

        "source_sha256": item["source_hash"],
        "fake_sha256": frozen_hash,

        "source_duration_s": f"{src['duration']:.6f}",
        "fake_duration_s": f"{out['duration']:.6f}",
        "duration_ratio": f"{item['duration_ratio']:.6f}",

        "sample_rate": out["sample_rate"],
        "channels": out["channels"],
        "peak": f"{out['peak']:.8f}",
        "rms": f"{out['rms']:.8f}",
        "clipped_fraction": f"{out['clipped_fraction']:.8f}",

        "rvc_version": "v2",
        "checkpoint_sr": "40k",
        "checkpoint_f0": 0,
        "model_info": "200epoch",

        "pitch": 0,
        "index_rate": 0.75,
        "rms_mix_rate": 1.0,
        "protect": 0.33,

        "selection_version": row["selection_version"],
    })

    manifest_files.append({
        "file": destination_name,
        "sha256": frozen_hash,
        "bytes": destination.stat().st_size,
        "source_file": row["original_file"],
        "source_speaker": row["speaker"],
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
        fieldnames=list(metadata_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(metadata_rows)


# =====================================================================
# MANIFEST
# =====================================================================

manifest = {
    "name": "External final fake holdout - RVC",
    "status": "FROZEN",

    "frozen_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "generator": "RVC",
    "count": len(manifest_files),

    "source_dataset": "external_final_voxpopuli",
    "source_speakers": sorted({
        x["selection"]["speaker"]
        for x in validated
    }),

    "model": {
        "name": "Mario_40",
        "path_at_generation": str(MODEL),
        "sha256": sha256(MODEL),
        "info": "200epoch",
        "version": "v2",
        "sample_rate": "40k",
        "f0": 0,
    },

    "index": {
        "path_at_generation": str(INDEX),
        "sha256": sha256(INDEX),
    },

    "parameters": PARAMETERS,

    "selection_csv": {
        "path_at_generation": str(SELECTION_CSV),
        "sha256": sha256(SELECTION_CSV),
    },

    "files": manifest_files,
}


with MANIFEST_JSON.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# =====================================================================
# RESULTADO
# =====================================================================

print()
print("=" * 72)
print("RVC EXTERNAL FINAL HOLDOUT — CONGELADO")
print("=" * 72)

print(f"Generator : RVC")
print(f"Model     : Mario_40")
print(f"Speakers  : {len(manifest['source_speakers'])}")
print(f"Audios    : {len(manifest_files)}")
print()
print(f"WAVs      : {FINAL_DIR}")
print(f"Metadata  : {METADATA_CSV}")
print(f"Manifest  : {MANIFEST_JSON}")

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