from pathlib import Path
import csv
import hashlib
import shutil
import sys

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

REAL_DIR = ROOT / "external_final_voxpopuli" / "real"
METADATA = ROOT / "external_final_voxpopuli" / "metadata_voxpopuli_real.csv"

BUILD_DIR = ROOT / "_external_fake_build"
OUT_DIR = BUILD_DIR / "rvc_sources"
OUT_CSV = BUILD_DIR / "rvc_selection.csv"

N_PER_SPEAKER = 2
EXPECTED_SPEAKERS = 10
SELECTION_VERSION = "RVC_EXTERNAL_V1"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_column(fieldnames, candidates):
    normalized = {x.lower().strip(): x for x in fieldnames}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


if not REAL_DIR.exists():
    sys.exit(f"ERROR: No existe {REAL_DIR}")

if not METADATA.exists():
    sys.exit(f"ERROR: No existe {METADATA}")

if OUT_DIR.exists() or OUT_CSV.exists():
    sys.exit(
        "\nERROR: La selección RVC ya existe.\n"
        "No se sobrescribe automáticamente para evitar cambiar el holdout.\n"
        f"Directorio: {OUT_DIR}\n"
        f"CSV: {OUT_CSV}\n"
    )


# ---------------------------------------------------------------------
# Índice de WAV congelados
# ---------------------------------------------------------------------

wav_files = list(REAL_DIR.rglob("*.wav"))

if not wav_files:
    sys.exit("ERROR: No hay WAVs en el holdout VoxPopuli.")

wav_by_name = {}

for wav in wav_files:
    wav_by_name.setdefault(wav.name, []).append(wav)


# ---------------------------------------------------------------------
# Leer metadata
# ---------------------------------------------------------------------

with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    if not reader.fieldnames:
        sys.exit("ERROR: El CSV no tiene cabecera.")

    speaker_col = find_column(
        reader.fieldnames,
        [
            "speaker",
            "speaker_id",
            "speakerid",
            "client_id",
            "clientid"
        ]
    )

    file_col = find_column(
        reader.fieldnames,
        [
            "file",
            "filename",
            "wav",
            "wav_file",
            "path",
            "audio",
            "audio_path"
        ]
    )

    if speaker_col is None:
        print("Columnas encontradas:", reader.fieldnames)
        sys.exit("ERROR: No encuentro la columna de speaker.")

    if file_col is None:
        print("Columnas encontradas:", reader.fieldnames)
        sys.exit("ERROR: No encuentro la columna que identifica el WAV.")

    rows = list(reader)


# ---------------------------------------------------------------------
# Resolver WAV de cada fila
# ---------------------------------------------------------------------

resolved = []

for row in rows:
    speaker = str(row[speaker_col]).strip()
    raw_file = str(row[file_col]).strip()

    if not speaker or not raw_file:
        continue

    name = Path(raw_file).name

    matches = wav_by_name.get(name, [])

    if len(matches) == 0:
        print(f"AVISO: no encontrado: {raw_file}")
        continue

    if len(matches) > 1:
        sys.exit(
            f"ERROR: Hay varios WAV con el mismo nombre: {name}"
        )

    resolved.append({
        "speaker": speaker,
        "wav": matches[0],
        "original_metadata": row
    })


# ---------------------------------------------------------------------
# Agrupar por speaker
# ---------------------------------------------------------------------

by_speaker = {}

for item in resolved:
    by_speaker.setdefault(item["speaker"], []).append(item)

speakers = sorted(by_speaker.keys())

if len(speakers) != EXPECTED_SPEAKERS:
    sys.exit(
        f"ERROR: esperaba {EXPECTED_SPEAKERS} speakers "
        f"pero he encontrado {len(speakers)}: {speakers}"
    )


# ---------------------------------------------------------------------
# Selección determinista
#
# No usamos random.sample().
# El orden depende de SHA256 y queda reproducible.
# ---------------------------------------------------------------------

selected = []

for speaker in speakers:

    candidates = by_speaker[speaker]

    if len(candidates) < N_PER_SPEAKER:
        sys.exit(
            f"ERROR: speaker {speaker} solo tiene "
            f"{len(candidates)} audios."
        )

    def deterministic_key(item):
        value = (
            f"{SELECTION_VERSION}|"
            f"{speaker}|"
            f"{item['wav'].name}"
        )

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    candidates = sorted(
        candidates,
        key=deterministic_key
    )

    chosen = candidates[:N_PER_SPEAKER]

    for index, item in enumerate(chosen, start=1):
        item = dict(item)
        item["speaker_index"] = index
        selected.append(item)


# ---------------------------------------------------------------------
# Copiar a staging
# ---------------------------------------------------------------------

BUILD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=False)

selection_rows = []

speaker_number = {
    speaker: i
    for i, speaker in enumerate(speakers, start=1)
}

for item in selected:

    speaker = item["speaker"]
    src = item["wav"]

    spk_num = speaker_number[speaker]
    num = item["speaker_index"]

    dst_name = (
        f"spk{spk_num:02d}_{num:02d}__{src.name}"
    )

    dst = OUT_DIR / dst_name

    shutil.copy2(src, dst)

    selection_rows.append({
        "rvc_input": dst_name,
        "original_file": src.name,
        "speaker": speaker,
        "source_sha256": sha256(src),
        "staged_sha256": sha256(dst),
        "target_voice": "Mario_40",
        "selection_version": SELECTION_VERSION
    })


# ---------------------------------------------------------------------
# Guardar selección
# ---------------------------------------------------------------------

with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rvc_input",
            "original_file",
            "speaker",
            "source_sha256",
            "staged_sha256",
            "target_voice",
            "selection_version"
        ]
    )

    writer.writeheader()
    writer.writerows(selection_rows)


# ---------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("RVC EXTERNAL SOURCES — PREPARADOS")
print("=" * 72)

print(f"Speakers       : {len(speakers)}")
print(f"Audios/speaker : {N_PER_SPEAKER}")
print(f"Total          : {len(selected)}")
print(f"Target RVC     : Mario_40")
print()
print(f"Sources : {OUT_DIR}")
print(f"CSV     : {OUT_CSV}")
print()

for speaker in speakers:
    n = sum(
        x["speaker"] == speaker
        for x in selection_rows
    )

    print(f"{speaker}: {n}")

print()
print("Los WAV originales de VoxPopuli NO han sido modificados.")
print("=" * 72)