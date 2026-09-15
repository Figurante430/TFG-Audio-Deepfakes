from pathlib import Path
import csv
import hashlib
import sys


ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

QWEN_JOBS = ROOT / "_external_fake_build" / "qwen_jobs.csv"
OUT_CSV = ROOT / "_external_fake_build" / "cosyvoice_jobs.csv"

EXPECTED = 20


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


if not QWEN_JOBS.exists():
    sys.exit(f"ERROR: No existe {QWEN_JOBS}")

if OUT_CSV.exists():
    sys.exit(
        f"ERROR: Ya existe {OUT_CSV}\n"
        "No se sobrescribe."
    )


with QWEN_JOBS.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    qwen_jobs = list(csv.DictReader(f))


if len(qwen_jobs) != EXPECTED:
    sys.exit(
        f"ERROR: esperaba {EXPECTED} jobs Qwen "
        f"pero hay {len(qwen_jobs)}."
    )


jobs = []

for i, row in enumerate(qwen_jobs, start=1):

    jobs.append({
        "id": f"cosyvoice_{i:03d}",
        "speaker": row["speaker"],

        # MISMA referencia externa que Qwen
        "reference_file": row["reference_file"],
        "original_reference_file": row["original_reference_file"],
        "reference_text": row["reference_text"],

        # MISMO texto objetivo que Qwen
        "target_text": row["target_text"],

        "language": "Spanish",
        "clone_mode": "zero_shot",

        # Namespace distinto, pero semillas deterministas
        "seed": 2026091601 + (i - 1),

        "target_text_sha256":
            sha256_text(row["target_text"]),
    })


speakers = sorted({
    x["speaker"]
    for x in jobs
})

if len(speakers) != 10:
    sys.exit(
        f"ERROR: esperaba 10 speakers, "
        f"hay {len(speakers)}."
    )


for speaker in speakers:
    n = sum(
        x["speaker"] == speaker
        for x in jobs
    )

    if n != 2:
        sys.exit(
            f"ERROR: {speaker} tiene {n} jobs."
        )


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
print("COSYVOICE EXTERNAL JOBS — PREPARADOS")
print("=" * 72)
print("Clone mode : zero_shot")
print(f"Speakers   : {len(speakers)}")
print(f"Jobs       : {len(jobs)}")
print()
print(f"CSV        : {OUT_CSV}")
print()
print("Referencias y textos son idénticos a los usados con Qwen.")
print("NO se ha generado audio todavía.")
print("=" * 72)