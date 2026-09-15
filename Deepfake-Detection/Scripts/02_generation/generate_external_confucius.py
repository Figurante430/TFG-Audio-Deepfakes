from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import random
import sys
import os

import numpy as np
import soundfile as sf
import torch


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

CONF_ROOT = Path(
    r"C:\Users\gonza\TFG\Confucius4-TTS"
)

CONFIG = (
    CONF_ROOT
    / "config"
    / "inference_config.yaml"
)

JOBS_CSV = (
    ROOT
    / "_external_fake_build"
    / "confucius_jobs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "_external_fake_build"
    / "confucius_outputs"
)

RESULTS_CSV = (
    ROOT
    / "_external_fake_build"
    / "confucius_generation.csv"
)

GEN_MANIFEST = (
    ROOT
    / "_external_fake_build"
    / "confucius_generation_manifest.json"
)

EXPECTED_JOBS = 20
EXPECTED_SPEAKERS = 10


# =====================================================================
# IMPORT LOCAL
# =====================================================================

sys.path.insert(
    0,
    str(CONF_ROOT)
)

from confuciustts.cli.inference import ConfuciusTTS


# =====================================================================
# UTILS
# =====================================================================

def fail(message):
    print()
    print("ERROR:", message)
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def inspect_audio(path: Path):

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    if audio.size == 0:
        raise RuntimeError("Audio vacío")

    if not np.isfinite(audio).all():
        raise RuntimeError(
            "Audio contiene NaN o Inf"
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

    duration = (
        audio.shape[0]
        / sr
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

if not torch.cuda.is_available():
    fail(
        "CUDA no está disponible "
        "en confucius-env."
    )


for path in [
    CONF_ROOT,
    CONFIG,
    JOBS_CSV,
]:
    if not path.exists():
        fail(
            f"No existe:\n{path}"
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


# =====================================================================
# LOAD MODEL
# =====================================================================

print()
print("=" * 72)
print("CARGANDO CONFUCIUS4-TTS")
print("=" * 72)

print(
    f"GPU       : "
    f"{torch.cuda.get_device_name(0)}"
)

print(f"Config    : {CONFIG}")
print("Language  : es")
print("Mode      : transcript-free zero-shot")
print()

# Confucius4 usa rutas relativas ./checkpoints/... dentro de su config.
# Por tanto, durante la carga el working directory debe ser la raíz
# del repositorio.
original_cwd = Path.cwd()
os.chdir(CONF_ROOT)

try:
    model = ConfuciusTTS(
        config_path=str(CONFIG),
        device="cuda",
    )
finally:
    os.chdir(original_cwd)


sample_rate = int(
    model.sample_rate
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
print("CONFUCIUS4 EXTERNAL GENERATION")
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


    target_text = (
        row["target_text"]
        .strip()
    )

    language = (
        row["language"]
        .strip()
    )

    seed = int(
        row["seed"]
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
        f"Speaker : {row['speaker']}"
    )

    print(
        f"Ref     : {reference_file.name}"
    )

    print(
        f"Lang    : {language}"
    )

    print(
        f"Seed    : {seed}"
    )


    # -------------------------------------------------------------
    # Seed fija
    # -------------------------------------------------------------

    seed_everything(seed)


    # -------------------------------------------------------------
    # Confucius4 transcript-free zero-shot
    #
    # IMPORTANTE:
    # NO se pasa reference_text.
    # -------------------------------------------------------------

    with torch.inference_mode():

        audio = model.generate(
            text=target_text,
            lang=language,
            prompt_wav=str(
                reference_file
            ),
            verbose=False,
        )


    if audio is None:
        fail(
            f"{job_id}: model.generate "
            "devolvió None."
        )


    # API oficial devuelve tensor de audio.
    if torch.is_tensor(audio):

        wav = (
            audio
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    else:

        wav = np.asarray(
            audio,
            dtype=np.float32
        )


    wav = np.squeeze(wav)


    if wav.ndim != 1:
        fail(
            f"{job_id}: shape inesperado "
            f"{wav.shape}"
        )


    if wav.size == 0:
        fail(
            f"{job_id}: audio vacío."
        )


    if not np.isfinite(wav).all():
        fail(
            f"{job_id}: audio con NaN/Inf."
        )


    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    sf.write(
        str(output_file),
        wav.astype(np.float32),
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
            f"{job_id}: prácticamente silencioso."
        )


    if info["duration"] > 30:
        print(
            f"AVISO: duración alta "
            f"({info['duration']:.2f}s)"
        )


    if info["clipped_fraction"] > 0.01:
        print(
            f"AVISO: clipping aproximado "
            f"{info['clipped_fraction'] * 100:.2f}%"
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
            "Confucius4-TTS",

        "speaker":
            row["speaker"],

        "reference_file":
            str(reference_file),

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        # Metadata únicamente.
        # NO se utilizó para generar.
        "reference_text":
            row["reference_text"],

        "reference_text_used":
            False,

        "target_text":
            target_text,

        "language":
            language,

        "clone_mode":
            "transcript_free_zero_shot",

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
# VALIDATE COUNT
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
# SAVE CSV
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
# IDENTIDAD DEL MODELO
# =====================================================================

model_files = {}


candidate_files = [
    CONFIG,
    CONF_ROOT / "checkpoints" / "t2s_model.safetensors",
    CONF_ROOT / "checkpoints" / "s2a_model.pt",
    CONF_ROOT / "checkpoints" / "wav2vec2bert_stats.pt",
]


for path in candidate_files:

    if path.exists():

        model_files[
            str(
                path.relative_to(
                    CONF_ROOT
                )
            )
        ] = {

            "bytes":
                path.stat().st_size,

            "sha256":
                sha256(path),
        }


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
        "Confucius4-TTS",

    "clone_mode":
        "transcript_free_zero_shot",

    "language":
        "es",

    "reference_transcript_used":
        False,

    "device":
        torch.cuda.get_device_name(0),

    "sample_rate":
        sample_rate,

    "config_path_at_generation":
        str(CONFIG),

    "model_files":
        model_files,

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
# END
# =====================================================================

print()
print("=" * 72)
print("CONFUCIUS4 EXTERNAL GENERATION — COMPLETADA")
print("=" * 72)

print("Generator : Confucius4-TTS")
print("Language  : es")
print("Mode      : transcript-free zero-shot")
print(f"Speakers  : {len(speakers)}")
print(f"Audios    : {len(results)}")
print(f"SR        : {sample_rate} Hz")

print()
print(f"WAVs      : {OUTPUT_DIR}")
print(f"Results   : {RESULTS_CSV}")
print(f"Manifest  : {GEN_MANIFEST}")

print()
print("Estado    : GENERADOS, TODAVÍA NO CONGELADOS")
print("=" * 72)