from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import random
import sys

import numpy as np
import soundfile as sf
import torch

from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

JOBS_CSV = (
    ROOT
    / "_external_fake_build"
    / "qwen_jobs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "_external_fake_build"
    / "qwen_outputs"
)

RESULTS_CSV = (
    ROOT
    / "_external_fake_build"
    / "qwen_generation.csv"
)

GENERATION_JSON = (
    ROOT
    / "_external_fake_build"
    / "qwen_generation_manifest.json"
)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

EXPECTED_JOBS = 20
EXPECTED_SPEAKERS = 10
EXPECTED_SR = 24000


# Parámetros oficiales de generation_config.json,
# fijados explícitamente para que no cambien si el paquete se actualiza.
GENERATION_PARAMS = {
    "do_sample": True,
    "repetition_penalty": 1.05,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,

    "subtalker_dosample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
    "subtalker_top_k": 50,

    "max_new_tokens": 8192,
}


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
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def inspect_wav(path: Path):

    wav, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    if wav.size == 0:
        raise RuntimeError("Audio vacío")

    if not np.isfinite(wav).all():
        raise RuntimeError(
            "Audio contiene NaN o Inf"
        )

    duration = len(wav) / sr

    peak = float(
        np.max(np.abs(wav))
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(wav)
            )
        )
    )

    return {
        "sample_rate": int(sr),
        "channels": int(wav.shape[1]),
        "duration": float(duration),
        "peak": peak,
        "rms": rms,
    }


# =====================================================================
# PRECHECK
# =====================================================================

if not JOBS_CSV.exists():
    fail(
        f"No existe:\n{JOBS_CSV}"
    )

if OUTPUT_DIR.exists():
    fail(
        "Ya existe qwen_outputs:\n"
        f"{OUTPUT_DIR}\n\n"
        "No se sobrescribe automáticamente."
    )

if RESULTS_CSV.exists():
    fail(
        f"Ya existe:\n{RESULTS_CSV}"
    )

if GENERATION_JSON.exists():
    fail(
        f"Ya existe:\n{GENERATION_JSON}"
    )

if not torch.cuda.is_available():
    fail(
        "CUDA no está disponible en este entorno."
    )


# =====================================================================
# JOBS
# =====================================================================

with JOBS_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    jobs = list(
        csv.DictReader(f)
    )


if len(jobs) != EXPECTED_JOBS:
    fail(
        f"Esperaba {EXPECTED_JOBS} jobs "
        f"pero hay {len(jobs)}."
    )


speakers = sorted({
    row["speaker"]
    for row in jobs
})


if len(speakers) != EXPECTED_SPEAKERS:
    fail(
        f"Esperaba {EXPECTED_SPEAKERS} speakers "
        f"pero hay {len(speakers)}."
    )


for speaker in speakers:

    n = sum(
        row["speaker"] == speaker
        for row in jobs
    )

    if n != 2:
        fail(
            f"{speaker} tiene {n} jobs; "
            "esperaba exactamente 2."
        )


# =====================================================================
# MODELO LOCAL
#
# Muy importante:
# local_files_only=True impide que descarguemos silenciosamente una
# revisión diferente justo antes del experimento final.
# =====================================================================

print()
print("Resolviendo snapshot local de Qwen...")

try:

    MODEL_PATH = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            local_files_only=True
        )
    )

except Exception as e:

    fail(
        "No encuentro el modelo completo en la caché local.\n"
        "No voy a descargar una revisión nueva automáticamente.\n\n"
        f"{e}"
    )


print(f"Modelo local: {MODEL_PATH}")

snapshot_revision = MODEL_PATH.name

print(f"Snapshot    : {snapshot_revision}")


# =====================================================================
# LOAD
# =====================================================================

print()
print("=" * 72)
print("CARGANDO QWEN3-TTS")
print("=" * 72)

print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"Model      : {MODEL_ID}")
print(f"Snapshot   : {snapshot_revision}")
print(f"Dtype      : bfloat16")
print(f"Attention  : SDPA")
print()


model = Qwen3TTSModel.from_pretrained(
    str(MODEL_PATH),

    device_map="cuda:0",
    dtype=torch.bfloat16,

    # Evitamos depender de FlashAttention.
    attn_implementation="sdpa",
)


# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False
)


results = []


# =====================================================================
# GENERACIÓN
# =====================================================================

print()
print("=" * 72)
print("QWEN EXTERNAL GENERATION")
print("=" * 72)


for index, row in enumerate(
    jobs,
    start=1
):

    job_id = row["id"]

    reference_file = Path(
        row["reference_file"]
    )

    if not reference_file.exists():
        fail(
            f"Falta referencia:\n"
            f"{reference_file}"
        )

    ref_text = row[
        "reference_text"
    ].strip()

    target_text = row[
        "target_text"
    ].strip()

    language = row[
        "language"
    ].strip()

    seed = int(
        row["seed"]
    )

    output_file = (
        OUTPUT_DIR
        / f"{job_id}.wav"
    )

    if output_file.exists():
        fail(
            f"Output inesperado ya existente:\n"
            f"{output_file}"
        )

    print()
    print(
        f"[{index:02d}/{len(jobs)}] "
        f"{job_id}"
    )

    print(
        f"Speaker : {row['speaker']}"
    )

    print(
        f"Ref     : {reference_file.name}"
    )

    print(
        f"Seed    : {seed}"
    )

    # -------------------------------------------------------------
    # Semilla fija
    # -------------------------------------------------------------

    seed_everything(seed)

    # -------------------------------------------------------------
    # Generación
    #
    # x_vector_only_mode=False =>
    # ICL utilizando audio + transcripción de referencia.
    # -------------------------------------------------------------

    wavs, sr = model.generate_voice_clone(

        text=target_text,

        language=language,

        ref_audio=str(
            reference_file
        ),

        ref_text=ref_text,

        x_vector_only_mode=False,

        # Queremos inferencia completa, no simulación streaming.
        non_streaming_mode=True,

        **GENERATION_PARAMS,
    )


    if len(wavs) != 1:
        fail(
            f"{job_id}: Qwen devolvió "
            f"{len(wavs)} audios."
        )


    if int(sr) != EXPECTED_SR:
        fail(
            f"{job_id}: sample rate "
            f"{sr} != {EXPECTED_SR}"
        )


    wav = np.asarray(
        wavs[0],
        dtype=np.float32
    )


    if wav.ndim > 1:
        wav = np.squeeze(wav)


    if wav.size == 0:
        fail(
            f"{job_id}: audio vacío."
        )


    if not np.isfinite(wav).all():
        fail(
            f"{job_id}: NaN/Inf."
        )


    # -------------------------------------------------------------
    # WRITE
    # -------------------------------------------------------------

    sf.write(
        str(output_file),
        wav,
        sr,
        subtype="PCM_16"
    )


    # -------------------------------------------------------------
    # QC
    # -------------------------------------------------------------

    info = inspect_wav(
        output_file
    )


    if info["duration"] < 0.5:
        fail(
            f"{job_id}: duración "
            f"demasiado corta "
            f"({info['duration']:.3f}s)"
        )


    if info["rms"] < 1e-5:
        fail(
            f"{job_id}: audio "
            f"prácticamente silencioso."
        )


    file_hash = sha256(
        output_file
    )


    results.append({

        "id": job_id,

        "file": output_file.name,

        "generator": "Qwen3-TTS",

        "model": MODEL_ID,

        "model_snapshot":
            snapshot_revision,

        "speaker":
            row["speaker"],

        "reference_file":
            row["reference_file"],

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        "reference_text":
            ref_text,

        "target_text":
            target_text,

        "language":
            language,

        "clone_mode":
            "ICL",

        "x_vector_only_mode":
            False,

        "seed":
            seed,

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

        "sha256":
            file_hash,
    })


    print(
        f"OK      : "
        f"{info['duration']:.2f}s | "
        f"{info['sample_rate']} Hz"
    )


# =====================================================================
# VALIDACIÓN FINAL
# =====================================================================

generated = sorted(
    OUTPUT_DIR.glob("*.wav")
)


if len(generated) != EXPECTED_JOBS:
    fail(
        f"Se generaron {len(generated)} WAV; "
        f"esperaba {EXPECTED_JOBS}."
    )


# =====================================================================
# CSV
# =====================================================================

with RESULTS_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# =====================================================================
# GENERATION MANIFEST
# =====================================================================

manifest = {

    "status": "GENERATED_NOT_FROZEN",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "generator":
        "Qwen3-TTS",

    "model_id":
        MODEL_ID,

    "model_snapshot":
        snapshot_revision,

    "model_path_at_generation":
        str(MODEL_PATH),

    "device":
        torch.cuda.get_device_name(0),

    "dtype":
        "torch.bfloat16",

    "attention":
        "sdpa",

    "language":
        "Spanish",

    "clone_mode":
        "ICL",

    "x_vector_only_mode":
        False,

    "non_streaming_mode":
        True,

    "generation_parameters":
        GENERATION_PARAMS,

    "jobs_csv": {
        "path": str(
            JOBS_CSV
        ),

        "sha256": sha256(
            JOBS_CSV
        ),
    },

    "count":
        len(results),

    "speakers":
        speakers,

    "files": [
        {
            "file":
                row["file"],

            "speaker":
                row["speaker"],

            "seed":
                row["seed"],

            "sha256":
                row["sha256"],
        }
        for row in results
    ],
}


with GENERATION_JSON.open(
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
# END
# =====================================================================

print()
print("=" * 72)
print("QWEN EXTERNAL GENERATION — COMPLETADA")
print("=" * 72)

print(f"Model    : {MODEL_ID}")
print(f"Snapshot : {snapshot_revision}")
print(f"Speakers : {len(speakers)}")
print(f"Audios   : {len(results)}")
print(f"SR       : {EXPECTED_SR} Hz")

print()
print(f"WAVs     : {OUTPUT_DIR}")
print(f"Results  : {RESULTS_CSV}")
print(f"Manifest : {GENERATION_JSON}")

print()
print("Estado   : GENERADOS, TODAVÍA NO CONGELADOS")
print("=" * 72)