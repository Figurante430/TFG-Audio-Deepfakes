from pathlib import Path
import csv
import hashlib
import sys


ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

QWEN_JOBS = (
    ROOT
    / "_external_fake_build"
    / "qwen_jobs.csv"
)

OUT_CSV = (
    ROOT
    / "_external_fake_build"
    / "voxcpm2_jobs.csv"
)

EXPECTED = 20


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


if not QWEN_JOBS.exists():
    sys.exit(
        f"ERROR: No existe {QWEN_JOBS}"
    )


if OUT_CSV.exists():
    sys.exit(
        f"ERROR: Ya existe {OUT_CSV}\n"
        "No se sobrescribe."
    )


# ============================================================
# REUTILIZAR EXACTAMENTE EL DISEÑO DEL HOLDOUT TTS
# ============================================================

with QWEN_JOBS.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    source_jobs = list(
        csv.DictReader(f)
    )


if len(source_jobs) != EXPECTED:
    sys.exit(
        f"ERROR: esperaba {EXPECTED} jobs, "
        f"pero hay {len(source_jobs)}."
    )


jobs = []


for i, row in enumerate(
    source_jobs,
    start=1
):

    jobs.append({

        "id":
            f"voxcpm2_{i:03d}",

        "speaker":
            row["speaker"],

        # MISMA referencia externa
        "reference_file":
            row["reference_file"],

        "original_reference_file":
            row["original_reference_file"],

        # Se conserva únicamente como metadata.
        # NO se pasa a VoxCPM2.
        "reference_text":
            row["reference_text"],

        # MISMO texto objetivo usado en todos los TTS
        "target_text":
            row["target_text"],

        "language":
            "Spanish",

        "clone_mode":
            "reference_only_voice_cloning",

        # Seeds deterministas propias de VoxCPM2
        "seed":
            2026091901 + (i - 1),

        "cfg_value":
            2.0,

        "inference_timesteps":
            10,

        "normalize":
            False,

        "denoise":
            False,

        "target_text_sha256":
            sha256_text(
                row["target_text"]
            ),
    })


# ============================================================
# BALANCE
# ============================================================

speakers = sorted({
    row["speaker"]
    for row in jobs
})


if len(speakers) != 10:
    sys.exit(
        f"ERROR: esperaba 10 speakers, "
        f"hay {len(speakers)}."
    )


for speaker in speakers:

    count = sum(
        row["speaker"] == speaker
        for row in jobs
    )

    if count != 2:
        sys.exit(
            f"ERROR: {speaker} tiene "
            f"{count} jobs."
        )


# ============================================================
# VALIDAR REFERENCIAS
# ============================================================

for row in jobs:

    ref = Path(
        row["reference_file"]
    )

    if not ref.exists():
        sys.exit(
            f"ERROR: falta referencia:\n{ref}"
        )


# ============================================================
# SAVE
# ============================================================

with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=jobs[0].keys()
    )

    writer.writeheader()
    writer.writerows(jobs)


print()
print("=" * 72)
print("VOXCPM2 EXTERNAL JOBS — PREPARADOS")
print("=" * 72)

print("Generator  : VoxCPM2")
print("Language   : Spanish")
print("Clone mode : reference-only voice cloning")
print("CFG        : 2.0")
print("Steps      : 10")
print("Normalize  : False")
print("Denoise    : False")

print(f"Speakers   : {len(speakers)}")
print(f"Jobs       : {len(jobs)}")

print()
print(f"CSV        : {OUT_CSV}")

print()
print(
    "Referencias y textos objetivo son idénticos "
    "a los usados en el resto del holdout TTS."
)

print(
    "reference_text se conserva como metadata, "
    "pero NO se pasará al modelo."
)

print("NO se ha generado audio todavía.")
print("=" * 72)