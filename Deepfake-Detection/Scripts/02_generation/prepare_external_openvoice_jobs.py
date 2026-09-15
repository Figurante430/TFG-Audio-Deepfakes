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


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

OV_ROOT = Path(
    r"C:\Users\gonza\TFG\OpenVoiceV2\OpenVoice"
)

CONVERTER_DIR = (
    OV_ROOT
    / "checkpoints_v2"
    / "converter"
)

CONVERTER_CONFIG = (
    CONVERTER_DIR
    / "config.json"
)

CONVERTER_CKPT = (
    CONVERTER_DIR
    / "checkpoint.pth"
)

JOBS_CSV = (
    ROOT
    / "_external_fake_build"
    / "openvoice_jobs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "_external_fake_build"
    / "openvoice_outputs"
)

WORK_DIR = (
    ROOT
    / "_external_fake_build"
    / "openvoice_work"
)

RESULTS_CSV = (
    ROOT
    / "_external_fake_build"
    / "openvoice_generation.csv"
)

GEN_MANIFEST = (
    ROOT
    / "_external_fake_build"
    / "openvoice_generation_manifest.json"
)

EXPECTED_JOBS = 20
EXPECTED_SPEAKERS = 10

MELO_LANGUAGE = "ES"
MELO_SPEAKER = "ES"
MELO_SPEED = 1.0

# Texto fijo únicamente para reconstruir el embedding
# de la voz fuente MeloTTS española.
CALIBRATION_TEXT = (
    "Esta grabación de calibración se utiliza exclusivamente "
    "para obtener una representación estable de la voz española "
    "empleada como fuente antes de realizar la conversión de timbre."
)


# =====================================================================
# LOCAL IMPORTS
# =====================================================================

sys.path.insert(
    0,
    str(OV_ROOT)
)

from openvoice.api import ToneColorConverter
from melo.api import TTS


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

    clipped_fraction = float(
        np.mean(
            np.abs(audio) >= 0.999
        )
    )

    duration = (
        audio.shape[0]
        / sr
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

for path in [
    OV_ROOT,
    CONVERTER_CONFIG,
    CONVERTER_CKPT,
    JOBS_CSV,
]:
    if not path.exists():
        fail(
            f"No existe:\n{path}"
        )


for path in [
    OUTPUT_DIR,
    WORK_DIR,
    RESULTS_CSV,
    GEN_MANIFEST,
]:
    if path.exists():
        fail(
            f"Ya existe:\n{path}\n"
            "No se sobrescribe automáticamente."
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

    count = sum(
        row["speaker"] == speaker
        for row in jobs
    )

    if count != 2:
        fail(
            f"{speaker}: {count} jobs; "
            "esperaba 2."
        )


# =====================================================================
# DEVICE
# =====================================================================

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 72)
print("CARGANDO OPENVOICE V2")
print("=" * 72)

print(f"Device    : {device}")
print(f"Converter : {CONVERTER_CKPT}")
print("Base TTS  : MeloTTS ES")
print("Mode      : tone color conversion")
print()


# =====================================================================
# LOAD CONVERTER
# =====================================================================

tone_color_converter = ToneColorConverter(
    str(CONVERTER_CONFIG),
    device=device
)

tone_color_converter.load_ckpt(
    str(CONVERTER_CKPT)
)


# =====================================================================
# LOAD MELOTTS
# =====================================================================

melo_model = TTS(
    language=MELO_LANGUAGE,
    device=device
)

speaker_ids = (
    melo_model
    .hps
    .data
    .spk2id
)


print(
    "Melo speakers:",
    speaker_ids
)


if MELO_SPEAKER not in speaker_ids:
    fail(
        f"MeloTTS no contiene speaker '{MELO_SPEAKER}'.\n"
        f"Disponibles: {speaker_ids}"
    )


melo_speaker_id = (
    speaker_ids[
        MELO_SPEAKER
    ]
)


# =====================================================================
# WORK DIR
# =====================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False
)

WORK_DIR.mkdir(
    parents=True,
    exist_ok=False
)


# =====================================================================
# RECONSTRUIR SOURCE_SE
#
# El bundle V2 puede no incluir base_speakers/ses.
# Generamos una muestra fija de la voz MeloTTS ES y extraemos
# su embedding con el mismo converter.
# =====================================================================

print()
print("=" * 72)
print("EXTRAYENDO EMBEDDING DE VOZ FUENTE MELOTTS")
print("=" * 72)


calibration_wav = (
    WORK_DIR
    / "melo_es_calibration.wav"
)


seed_everything(
    2026091900
)


melo_model.tts_to_file(
    CALIBRATION_TEXT,
    melo_speaker_id,
    str(calibration_wav),
    speed=MELO_SPEED,
)


if not calibration_wav.exists():
    fail(
        "MeloTTS no creó el audio "
        "de calibración."
    )


calibration_info = inspect_audio(
    calibration_wav
)


print(
    f"Calibration : "
    f"{calibration_info['duration']:.2f}s | "
    f"{calibration_info['sample_rate']} Hz"
)


source_se_dir = (
    WORK_DIR
    / "source_se"
)

source_se_dir.mkdir(
    parents=True,
    exist_ok=True
)


# vad=True es preferible aquí en Windows:
# evita depender de la segmentación Whisper.
source_se = tone_color_converter.extract_se(
    str(calibration_wav)
)


if source_se is None:
    fail(
        "No se pudo extraer source_se."
    )


print(
    "source_se OK:",
    tuple(source_se.shape)
)


# =====================================================================
# GENERATION
# =====================================================================

results = []


print()
print("=" * 72)
print("OPENVOICE V2 EXTERNAL GENERATION")
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

    seed = int(
        row["seed"]
    )


    base_file = (
        WORK_DIR
        / f"{job_id}_melo.wav"
    )

    final_file = (
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
        f"Seed    : {seed}"
    )


    # -------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------

    seed_everything(
        seed
    )


    # -------------------------------------------------------------
    # PASO 1 — MeloTTS español
    # -------------------------------------------------------------

    melo_model.tts_to_file(
        target_text,
        melo_speaker_id,
        str(base_file),
        speed=MELO_SPEED,
    )


    if not base_file.exists():
        fail(
            f"{job_id}: MeloTTS "
            "no produjo audio."
        )


    # -------------------------------------------------------------
    # PASO 2 — embedding de la referencia VoxPopuli
    # -------------------------------------------------------------

    target_se = tone_color_converter.extract_se(
    str(reference_file)
)


    if target_se is None:
        fail(
            f"{job_id}: no se pudo "
            "extraer target_se."
        )


    # -------------------------------------------------------------
    # PASO 3 — conversión de color de voz
    # -------------------------------------------------------------

    tone_color_converter.convert(
        audio_src_path=str(
            base_file
        ),

        src_se=source_se,

        tgt_se=target_se,

        output_path=str(
            final_file
        ),

        # Watermark/message fijo.
        message="@MyShell",
    )


    if not final_file.exists():
        fail(
            f"{job_id}: OpenVoice "
            "no produjo output."
        )


    # -------------------------------------------------------------
    # QC
    # -------------------------------------------------------------

    base_info = inspect_audio(
        base_file
    )

    info = inspect_audio(
        final_file
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
            "prácticamente silencioso."
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


    output_hash = sha256(
        final_file
    )


    results.append({

        "id":
            job_id,

        "file":
            final_file.name,

        "generator":
            "OpenVoice V2",

        "base_generator":
            "MeloTTS",

        "base_language":
            MELO_LANGUAGE,

        "base_speaker":
            MELO_SPEAKER,

        "base_speed":
            MELO_SPEED,

        "speaker":
            row["speaker"],

        "reference_file":
            str(reference_file),

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        "reference_text":
            row["reference_text"],

        "reference_text_used":
            False,

        "target_text":
            target_text,

        "language":
            "ES",

        "clone_mode":
            "melo_tts_plus_tone_color_conversion",

        "seed":
            seed,

        "base_sample_rate":
            base_info[
                "sample_rate"
            ],

        "sample_rate":
            info[
                "sample_rate"
            ],

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
            output_hash,
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
# CSV
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
# MANIFEST
# =====================================================================

manifest = {

    "status":
        "GENERATED_NOT_FROZEN",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "generator":
        "OpenVoice V2",

    "converter": {

        "config":
            str(
                CONVERTER_CONFIG
            ),

        "config_sha256":
            sha256(
                CONVERTER_CONFIG
            ),

        "checkpoint":
            str(
                CONVERTER_CKPT
            ),

        "checkpoint_sha256":
            sha256(
                CONVERTER_CKPT
            ),
    },

    "base_tts": {

        "generator":
            "MeloTTS",

        "language":
            MELO_LANGUAGE,

        "speaker":
            MELO_SPEAKER,

        "speed":
            MELO_SPEED,

        "source_se_strategy":
            "direct_full_wav_embedding_from_fixed_calibration_audio",

        "calibration_text":
            CALIBRATION_TEXT,

        "calibration_wav_sha256":
            sha256(
                calibration_wav
            ),
    },

    "target_se_strategy":
        "ToneColorConverter.extract_se direct full reference WAV",

    "reference_transcript_used":
        False,

    "device":
        device,

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

sample_rates = sorted({
    int(x["sample_rate"])
    for x in results
})


print()
print("=" * 72)
print("OPENVOICE V2 EXTERNAL GENERATION — COMPLETADA")
print("=" * 72)

print("Generator : OpenVoice V2")
print("Base TTS  : MeloTTS ES")
print("Mode      : tone color conversion")
print(f"Speakers  : {len(speakers)}")
print(f"Audios    : {len(results)}")
print(
    "SR        : "
    + ", ".join(
        str(x)
        for x in sample_rates
    )
    + " Hz"
)

print()
print(f"WAVs      : {OUTPUT_DIR}")
print(f"Results   : {RESULTS_CSV}")
print(f"Manifest  : {GEN_MANIFEST}")

print()
print("Estado    : GENERADOS, TODAVÍA NO CONGELADOS")
print("=" * 72)