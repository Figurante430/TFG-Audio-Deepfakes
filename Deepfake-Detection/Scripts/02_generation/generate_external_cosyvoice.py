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

COSY_ROOT = Path(
    r"C:\Users\gonza\TFG\CosyVoice\CosyVoice"
)

MODEL_DIR = (
    COSY_ROOT
    / "pretrained_models"
    / "Fun-CosyVoice3-0.5B"
)

JOBS_CSV = (
    ROOT
    / "_external_fake_build"
    / "cosyvoice_jobs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "_external_fake_build"
    / "cosyvoice_outputs"
)

RESULTS_CSV = (
    ROOT
    / "_external_fake_build"
    / "cosyvoice_generation.csv"
)

GEN_MANIFEST = (
    ROOT
    / "_external_fake_build"
    / "cosyvoice_generation_manifest.json"
)

EXPECTED_JOBS = 20
EXPECTED_SPEAKERS = 10

PROMPT_PREFIX = (
    "You are a helpful assistant.<|endofprompt|>"
)


# =====================================================================
# IMPORT COSYVOICE LOCAL
# =====================================================================

sys.path.insert(
    0,
    str(COSY_ROOT)
)

MATCHA = (
    COSY_ROOT
    / "third_party"
    / "Matcha-TTS"
)

sys.path.insert(
    0,
    str(MATCHA)
)

from cosyvoice.cli.cosyvoice import AutoModel


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
        raise RuntimeError("NaN/Inf en audio")

    peak = float(
        np.max(np.abs(audio))
    )

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
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
    }


# =====================================================================
# PRECHECK
# =====================================================================

if not torch.cuda.is_available():
    fail(
        "CUDA no está disponible "
        "en el entorno CosyVoice."
    )


for path in [
    MODEL_DIR,
    JOBS_CSV,
]:
    if not path.exists():
        fail(
            f"No existe:\n{path}"
        )


# Archivos esenciales de CosyVoice3
essential_model_files = [
    MODEL_DIR / "cosyvoice3.yaml",
    MODEL_DIR / "llm.pt",
    MODEL_DIR / "flow.pt",
    MODEL_DIR / "hift.pt",
    MODEL_DIR / "campplus.onnx",
    MODEL_DIR / "speech_tokenizer_v3.onnx",
]


for path in essential_model_files:
    if not path.exists():
        fail(
            f"Falta archivo del modelo:\n{path}"
        )


for path in [
    OUTPUT_DIR,
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
# LOAD MODEL
# =====================================================================

print()
print("=" * 72)
print("CARGANDO FUN-COSYVOICE3")
print("=" * 72)

print(
    f"GPU       : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"Model     : {MODEL_DIR}"
)

print("Mode      : zero-shot")
print("Streaming : False")
print("Speed     : 1.0")
print("FP16      : True")
print()


cosyvoice = AutoModel(
    model_dir=str(MODEL_DIR),
    load_trt=False,
    load_vllm=False,
    fp16=True,
)


sample_rate = int(
    cosyvoice.sample_rate
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
# GENERATION
# =====================================================================

print()
print("=" * 72)
print("COSYVOICE EXTERNAL GENERATION")
print("=" * 72)


for index, row in enumerate(
    jobs,
    start=1
):

    job_id = row["id"]

    ref_file = Path(
        row["reference_file"]
    )

    if not ref_file.exists():
        fail(
            f"Falta referencia:\n{ref_file}"
        )


    ref_text = (
        row["reference_text"]
        .strip()
    )

    target_text = (
        row["target_text"]
        .strip()
    )

    seed = int(
        row["seed"]
    )


    # --------------------------------------------------------------
    # CosyVoice3 necesita este formato de prompt
    # --------------------------------------------------------------

    cosy_prompt_text = (
        PROMPT_PREFIX
        + ref_text
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
        f"Ref     : {ref_file.name}"
    )

    print(
        f"Seed    : {seed}"
    )


    # --------------------------------------------------------------
    # Seed fijo
    # --------------------------------------------------------------

    seed_everything(seed)


    # --------------------------------------------------------------
    # Zero-shot
    #
    # CosyVoice puede devolver más de un chunk si el frontend divide
    # el texto. Los concatenamos SIN ningún postprocesado adicional.
    # --------------------------------------------------------------

    chunks = []


    generator = cosyvoice.inference_zero_shot(
        target_text,
        cosy_prompt_text,
        str(ref_file),

        stream=False,
        speed=1.0,
    )


    for result in generator:

        if "tts_speech" not in result:
            fail(
                f"{job_id}: resultado "
                "sin tts_speech."
            )

        chunk = (
            result["tts_speech"]
            .detach()
            .cpu()
        )

        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)

        chunks.append(chunk)


    if not chunks:
        fail(
            f"{job_id}: no se generó audio."
        )


    speech = torch.cat(
        chunks,
        dim=1
    )


    wav = (
        speech
        .numpy()
        .flatten()
        .astype(np.float32)
    )


    if wav.size == 0:
        fail(
            f"{job_id}: audio vacío."
        )


    if not np.isfinite(wav).all():
        fail(
            f"{job_id}: NaN/Inf."
        )


    # --------------------------------------------------------------
    # Guardar PCM16
    # --------------------------------------------------------------

    sf.write(
        str(output_file),
        wav,
        sample_rate,
        subtype="PCM_16"
    )


    # --------------------------------------------------------------
    # QC
    # --------------------------------------------------------------

    info = inspect_audio(
        output_file
    )


    if info["duration"] < 0.5:
        fail(
            f"{job_id}: duración "
            f"demasiado corta "
            f"({info['duration']:.3f}s)"
        )


    if info["duration"] > 30:
        print(
            "AVISO: duración alta: "
            f"{info['duration']:.2f}s"
        )


    if info["rms"] < 1e-5:
        fail(
            f"{job_id}: prácticamente silencioso."
        )


    output_hash = sha256(
        output_file
    )


    results.append({

        "id":
            job_id,

        "file":
            output_file.name,

        "generator":
            "Fun-CosyVoice3",

        "model":
            "Fun-CosyVoice3-0.5B",

        "model_path":
            str(MODEL_DIR),

        "speaker":
            row["speaker"],

        "reference_file":
            str(ref_file),

        "original_reference_file":
            row[
                "original_reference_file"
            ],

        "reference_text":
            ref_text,

        "cosyvoice_prompt_text":
            cosy_prompt_text,

        "target_text":
            target_text,

        "language":
            "Spanish",

        "clone_mode":
            "zero_shot",

        "seed":
            seed,

        "stream":
            False,

        "speed":
            1.0,

        "fp16":
            True,

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
            output_hash,
    })


    print(
        f"OK      : "
        f"{info['duration']:.2f}s | "
        f"{info['sample_rate']} Hz"
    )


# =====================================================================
# FINAL VALIDATION
# =====================================================================

generated_files = sorted(
    OUTPUT_DIR.glob("*.wav")
)


if len(generated_files) != EXPECTED_JOBS:
    fail(
        f"Hay {len(generated_files)} WAV; "
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
    writer.writerows(
        results
    )


# =====================================================================
# MODEL IDENTITY
# =====================================================================

model_files = {}


for path in essential_model_files:

    model_files[
        path.name
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
        "Fun-CosyVoice3",

    "model":
        "Fun-CosyVoice3-0.5B",

    "model_path_at_generation":
        str(MODEL_DIR),

    "model_files":
        model_files,

    "device":
        torch.cuda.get_device_name(0),

    "sample_rate":
        sample_rate,

    "clone_mode":
        "zero_shot",

    "language":
        "Spanish",

    "prompt_prefix":
        PROMPT_PREFIX,

    "stream":
        False,

    "speed":
        1.0,

    "fp16":
        True,

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
print("COSYVOICE EXTERNAL GENERATION — COMPLETADA")
print("=" * 72)

print("Model    : Fun-CosyVoice3-0.5B")
print("Mode     : zero_shot")
print(f"Speakers : {len(speakers)}")
print(f"Audios   : {len(results)}")
print(f"SR       : {sample_rate} Hz")

print()
print(f"WAVs     : {OUTPUT_DIR}")
print(f"Results  : {RESULTS_CSV}")
print(f"Manifest : {GEN_MANIFEST}")

print()
print("Estado   : GENERADOS, TODAVÍA NO CONGELADOS")
print("=" * 72)