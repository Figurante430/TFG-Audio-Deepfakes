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
    / "confucius_jobs.csv"
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
# REUTILIZAR EXACTAMENTE EL DISEÑO QWEN
# ============================================================

with QWEN_JOBS.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    qwen_jobs = list(
        csv.DictReader(f)
    )


if len(qwen_jobs) != EXPECTED:
    sys.exit(
        f"ERROR: esperaba {EXPECTED} jobs Qwen "
        f"pero hay {len(qwen_jobs)}."
    )


jobs = []


for i, row in enumerate(
    qwen_jobs,
    start=1
):

    jobs.append({

        "id":
            f"confucius_{i:03d}",

        "speaker":
            row["speaker"],

        # MISMA voz externa
        "reference_file":
            row["reference_file"],

        "original_reference_file":
            row["original_reference_file"],

        # Se conserva solo como metadata.
        # Confucius4 NO la recibe en inferencia estándar.
        "reference_text":
            row["reference_text"],

        # MISMO texto objetivo que Qwen/CosyVoice
        "target_text":
            row["target_text"],

        # Código oficial para español
        "language":
            "es",

        "clone_mode":
            "transcript_free_zero_shot",

        # Seeds nuevos pero deterministas
        "seed":
            2026091701 + (i - 1),

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
print("CONFUCIUS4 EXTERNAL JOBS — PREPARADOS")
print("=" * 72)

print("Generator  : Confucius4-TTS")
print("Language   : es")
print("Clone mode : transcript-free zero-shot")
print(f"Speakers   : {len(speakers)}")
print(f"Jobs       : {len(jobs)}")

print()
print(f"CSV        : {OUT_CSV}")

print()
print(
    "Referencias y textos objetivo son idénticos "
    "a Qwen/CosyVoice."
)

print(
    "reference_text se conserva como metadata, "
    "pero NO se pasará al modelo."
)

print("NO se ha generado audio todavía.")
print("=" * 72)