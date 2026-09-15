from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

import numpy as np
import soundfile as sf
import torch
import random


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
VOX_ROOT = Path(r"C:\Users\gonza\TFG\VoxCPM")

JOBS_CSV = (
    ROOT
    / "_external_fake_build"
    / "voxcpm2_jobs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "_external_fake_build"
    / "voxcpm2_outputs"
)

RESULTS_CSV = (
    ROOT
    / "_external_fake_build"
    / "voxcpm2_generation.csv"
)

GEN_MANIFEST = (
    ROOT
    / "_external_fake_build"
    / "voxcpm2_generation_manifest.json"
)

EXPECTED_JOBS = 20
EXPECTED_SPEAKERS = 10

MODEL_ID = "openbmb/VoxCPM2"


# =====================================================================
# IMPORT
# =====================================================================

from voxcpm import VoxCPM


# =====================================================================
# UTILS
# =====================================================================

def fail(msg):
    print()
    print("ERROR:", msg)
    sys.exit(1)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        raise RuntimeError("audio vacío")

    if not np.isfinite(audio).all():
        raise RuntimeError(
            "audio contiene NaN o Inf"
        )

    duration = (
        audio.shape[0]
        / sr
    )

    peak = float(
        np.max(np.abs(audio))
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
            np.abs(audio) >= 0.999
        )
    )

    return {
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "duration": float(duration),
        "peak": peak,
        "rms": rms,
        "clipped_fraction": clipped_fraction,
    }


# =====================================================================
# PRECHECK
# =====================================================================

if not JOBS_CSV.exists():
    fail(
        f"No existe:\n{JOBS_CSV}"
    )


for path in [
    OUTPUT_DIR,
    RESULTS_CSV,
    GEN_MANIFEST,
]:
    if path.exists():
        fail(
            f"Ya existe:\n{path}\n"
            "No se sobrescribe."
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
        f"Esperaba {EXPECTED_JOBS} jobs, "
        f"hay {len(jobs)}."
    )


speakers = sorted({
    row["speaker"]
    for row in jobs
})


if len(speakers) != EXPECTED_SPEAKERS:
    fail(
        f"Esperaba {EXPECTED_SPEAKERS} speakers, "
        f"hay {len(speakers)}."
    )


for speaker in speakers:

    n = sum(
        row["speaker"] == speaker
        for row in jobs
    )

    if n != 2:
        fail(
            f"{speaker}: {n} jobs; "
            "esperaba 2."
        )


for row in jobs:

    reference_file = Path(
        row["reference_file"]
    )

    if not reference_file.exists():
        fail(
            "Falta referencia:\n"
            f"{reference_file}"
        )


# =====================================================================
# DEVICE
# =====================================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 72)
print("CARGANDO VOXCPM2")
print("=" * 72)

print(f"Model     : {MODEL_ID}")
print(f"Device    : {device}")

if torch.cuda.is_available():
    print(
        "GPU       : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("Denoiser  : DISABLED")
print("Optimize  : False")
print()


# =====================================================================
# LOCAL MODEL RESOLUTION
#
# No queremos descargar silenciosamente una versión nueva.
# Primero buscamos instalaciones locales habituales.
# Si no existen, se permite SOLO el cache local de HuggingFace.
# =====================================================================

local_candidates = [

    VOX_ROOT
    / "pretrained_models"
    / "VoxCPM2",

    VOX_ROOT
    / "models"
    / "VoxCPM2",

    VOX_ROOT
    / "checkpoints"
    / "VoxCPM2",

    VOX_ROOT
    / "VoxCPM2",
]


model_source = None
model_source_type = None


for candidate in local_candidates:

    if candidate.is_dir():

        model_source = str(
            candidate.resolve()
        )

        model_source_type = (
            "local_directory"
        )

        break


if model_source is None:

    model_source = MODEL_ID
    model_source_type = (
        "huggingface_local_cache"
    )


print(
    f"Source    : {model_source}"
)

print(
    f"SourceType: {model_source_type}"
)

print()


# =====================================================================
# LOAD MODEL
# =====================================================================

try:

    model = VoxCPM.from_pretrained(
        model_source,
        load_denoiser=False,
        local_files_only=(
            model_source_type
            == "huggingface_local_cache"
        ),
        optimize=False,
        device=device,
    )

except Exception as exc:

    fail(
        "No se pudo cargar VoxCPM2 sin descargar "
        "un modelo nuevo.\n\n"
        f"{type(exc).__name__}: {exc}"
    )


sample_rate = int(
    model.tts_model.sample_rate
)


print(
    f"Sample rate: {sample_rate} Hz"
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
# GENERATE
# =====================================================================

print()
print("=" * 72)
print("VOXCPM2 EXTERNAL GENERATION")
print("=" * 72)


for index, row in enumerate(
    jobs,
    start=1
):

    job_id = (
        row["id"].strip()
    )

    speaker = (
        row["speaker"].strip()
    )

    reference_file = Path(
        row["reference_file"]
    )

    target_text = (
        row["target_text"]
        .strip()
    )

    seed = int(
        row["seed"]
    )

    cfg_value = float(
        row["cfg_value"]
    )

    inference_timesteps = int(
        row["inference_timesteps"]
    )

    normalize = (
        row["normalize"]
        .strip()
        .lower()
        == "true"
    )

    denoise = (
        row["denoise"]
        .strip()
        .lower()
        == "true"
    )


    output_file = (
        OUTPUT_DIR
        / f"{job_id}.wav"
    )


    print()
    print(
        f"[{index:02d}/{len(jobs)}] "
        f"{job_id}"
    )

    print(
        f"Speaker : {speaker}"
    )

    print(
        f"Ref     : {reference_file.name}"
    )

    print(
        f"Seed    : {seed}"
    )

    print(
        f"CFG     : {cfg_value}"
    )

    print(
        f"Steps   : {inference_timesteps}"
    )


    # -------------------------------------------------------------
    # VOXCPM2
    #
    # Reference-only mode:
    # - NO prompt_wav_path
    # - NO prompt_text
    # - reference_wav_path únicamente
    #
    # retry_badcase=False:
    # evita que el modelo cambie internamente de seed.
    # -------------------------------------------------------------

    # La versión local de VoxCPM no expone seed= en generate().
    # Fijamos los RNG globales antes de cada muestra.
    seed_everything(seed)
    wav = model.generate(
        text=target_text,

        reference_wav_path=str(
            reference_file
        ),

        cfg_value=cfg_value,

        inference_timesteps=
            inference_timesteps,

        normalize=normalize,

        denoise=denoise,

        retry_badcase=False,
    )


    if wav is None:
        fail(
            f"{job_id}: generate() "
            "devolvió None."
        )


    wav = np.asarray(
        wav,
        dtype=np.float32
    )

    wav = np.squeeze(
        wav
    )


    if wav.ndim != 1:
        fail(
            f"{job_id}: shape inesperado "
            f"{wav.shape}"
        )


    if wav.size == 0:
        fail(
            f"{job_id}: audio vacío."
        )


    if not np.isfinite(
        wav
    ).all():
        fail(
            f"{job_id}: audio "
            "contiene NaN/Inf."
        )


    # -------------------------------------------------------------
    # SAVE PCM16
    # -------------------------------------------------------------

    sf.write(
        str(output_file),
        wav,
        sample_rate,
        subtype="PCM_16"
    )


    # -------------------------------------------------------------
    # QC
    # -------------------------------------------------------------

    info = inspect_audio(
        output_file
    )


    if info["duration"] < 0.5:
        fail(
            f"{job_id}: duración demasiado corta "
            f"({info['duration']:.3f}s)"
        )


    if info["rms"] < 1e-5:
        fail(
            f"{job_id}: audio prácticamente "
            "silencioso."
        )


    if info["duration"] > 30:
        print(
            "AVISO: duración alta "
            f"{info['duration']:.2f}s"
        )


    if (
        info["clipped_fraction"]
        > 0.01
    ):
        print(
            "AVISO: clipping "
            f"{info['clipped_fraction']*100:.2f}%"
        )


    file_hash = sha256(
        output_file
    )


    results.append({

        "id":
            job_id,

        "file":
            output_file.name,

        "generator":
            "VoxCPM2",

        "model":
            MODEL_ID,

        "speaker":
            speaker,

        "reference_file":
            str(reference_file),

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        # Solo metadata.
        "reference_text":
            row["reference_text"],

        "reference_text_used":
            False,

        "target_text":
            target_text,

        "language":
            row["language"],

        "clone_mode":
            row["clone_mode"],

        "seed":
            seed,

        "cfg_value":
            cfg_value,

        "inference_timesteps":
            inference_timesteps,

        "normalize":
            normalize,

        "denoise":
            denoise,

        "retry_badcase":
            False,

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
            file_hash,
    })


    print(
        f"OK      : "
        f"{info['duration']:.2f}s | "
        f"{info['sample_rate']} Hz"
    )


# =====================================================================
# COUNT
# =====================================================================

generated = sorted(
    OUTPUT_DIR.glob("*.wav")
)


if len(generated) != EXPECTED_JOBS:
    fail(
        f"Hay {len(generated)} WAV; "
        f"esperaba {EXPECTED_JOBS}."
    )


# =====================================================================
# SAVE RESULTS
# =====================================================================

with RESULTS_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            results[0].keys()
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# =====================================================================
# MODEL IDENTITY
# =====================================================================

model_identity = {

    "requested_model_id":
        MODEL_ID,

    "source":
        model_source,

    "source_type":
        model_source_type,

    "sample_rate":
        sample_rate,
}


# Si utilizamos directorio local, registrar hashes
# de ficheros de configuración pequeños.
if (
    model_source_type
    == "local_directory"
):

    model_path = Path(
        model_source
    )

    identity_files = {}

    for name in [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ]:

        path = (
            model_path
            / name
        )

        if path.exists():

            identity_files[name] = {

                "bytes":
                    path.stat().st_size,

                "sha256":
                    sha256(path),
            }


    model_identity[
        "identity_files"
    ] = identity_files


# =====================================================================
# GENERATION MANIFEST
# =====================================================================

manifest = {

    "status":
        "GENERATED_NOT_FROZEN",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "generator":
        "VoxCPM2",

    "model_identity":
        model_identity,

    "clone_mode":
        "reference_only_voice_cloning",

    "reference_transcript_used":
        False,

    "generation_parameters": {

        "cfg_value":
            2.0,

        "inference_timesteps":
            10,

        "normalize":
            False,

        "denoise":
            False,

        "retry_badcase":
            False,

        "optimize":
            False,

        "seed_strategy":
            "global_python_numpy_torch_rng_before_each_generation",
    },

    "device":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),

    "sample_rate":
        sample_rate,

    "count":
        len(results),

    "speakers":
        speakers,

    "jobs_csv": {

        "path":
            str(JOBS_CSV),

        "sha256":
            sha256(JOBS_CSV),
    },

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


with GEN_MANIFEST.open(
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
# FINAL
# =====================================================================

print()
print("=" * 72)
print("VOXCPM2 EXTERNAL GENERATION — COMPLETADA")
print("=" * 72)

print("Generator : VoxCPM2")
print(f"Model     : {MODEL_ID}")
print("Mode      : reference-only voice cloning")
print(f"Speakers  : {len(speakers)}")
print(f"Audios    : {len(results)}")
print(f"SR        : {sample_rate} Hz")

print()
print(f"WAVs      : {OUTPUT_DIR}")
print(f"Results   : {RESULTS_CSV}")
print(f"Manifest  : {GEN_MANIFEST}")

print()
print(
    "Estado    : GENERADOS, TODAVÍA NO CONGELADOS"
)

print("=" * 72)