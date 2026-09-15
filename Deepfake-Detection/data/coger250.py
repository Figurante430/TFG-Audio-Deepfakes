from pathlib import Path
import shutil
import csv

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

SOURCE_DIR = ROOT / "data" / "tts" / "confucius4"
STAGING_DIR = ROOT / "data" / "sts" / "_sources_200"

# Coger WAV ordenados
wavs = sorted(
    [p for p in SOURCE_DIR.glob("*.wav") if p.is_file()],
    key=lambda p: p.name.lower()
)

if len(wavs) < 200:
    raise RuntimeError(
        f"Solo hay {len(wavs)} WAV en {SOURCE_DIR}"
    )

# Universo solicitado: primeros 250
pool = wavs[:250]

# De esos 250 utilizamos 200
selected = pool[:200]

STAGING_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for i, src in enumerate(selected, 1):

    dst = STAGING_DIR / f"{i:03d}_{src.name}"

    if not dst.exists():
        shutil.copy2(src, dst)

    rows.append({
        "id": i,
        "source_original": str(src),
        "source_staging": str(dst),
        "source_generator": "confucius4",
    })

metadata = STAGING_DIR / "sources.csv"

with metadata.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "source_original",
            "source_staging",
            "source_generator",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Preparados: {len(selected)} audios")
print(f"Carpeta: {STAGING_DIR}")
print(f"Metadata: {metadata}")