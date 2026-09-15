from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import wave
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
DATA_DIR = ROOT / "data"

METADATA_WITH_SPLIT = DATA_DIR / "metadata_with_split.csv"
METADATA_CSV = DATA_DIR / "metadata.csv"

OUTPUT_ROOT = ROOT / "data_normalized"
OUTPUT_METADATA = OUTPUT_ROOT / "metadata_normalized.csv"
ERRORS_CSV = OUTPUT_ROOT / "normalization_errors.csv"

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # PCM16 = 2 bytes


# ============================================================
# UTILIDADES
# ============================================================

def choose_metadata() -> Path:
    if METADATA_WITH_SPLIT.exists():
        return METADATA_WITH_SPLIT

    if METADATA_CSV.exists():
        return METADATA_CSV

    raise FileNotFoundError(
        "No encuentro metadata_with_split.csv ni metadata.csv en:\n"
        f"{DATA_DIR}"
    )


def load_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"{path.name} está vacío.")

    if "path" not in rows[0]:
        raise RuntimeError(
            f"{path.name} no contiene la columna 'path'."
        )

    return rows


def resolve_source(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return ROOT / path


def relative_to_data(source: Path) -> Path:
    """
    Convierte:
      C:\...\Deepfake-Detection\data\tts\qwen\001_ref1.wav
    en:
      tts\qwen\001_ref1.wav
    """
    try:
        return source.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise RuntimeError(
            f"El archivo no está dentro de {DATA_DIR}: {source}"
        )


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def is_valid_normalized_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False

    try:
        with wave.open(str(path), "rb") as wf:
            return (
                wf.getframerate() == TARGET_SR
                and wf.getnchannels() == TARGET_CHANNELS
                and wf.getsampwidth() == TARGET_SAMPLE_WIDTH
                and wf.getcomptype() == "NONE"
                and wf.getnframes() > 0
            )
    except Exception:
        return False


def convert_with_ffmpeg(source: Path, destination: Path):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),

        # Solo audio
        "-vn",

        # Mismo tratamiento para TODO el dataset
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le",

        str(destination),
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"FFmpeg terminó con código {completed.returncode}"
        )


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_dataset(rows, overwrite=False):
    normalized_rows = []
    errors = []

    total = len(rows)
    converted = 0
    skipped = 0

    for i, row in enumerate(rows, 1):
        source = resolve_source(row["path"])

        try:
            if not source.exists():
                raise FileNotFoundError(
                    f"No existe el archivo: {source}"
                )

            rel = relative_to_data(source)
            destination = OUTPUT_ROOT / rel

            print(
                f"[{i:04d}/{total}] "
                f"{source.name} -> {destination.relative_to(OUTPUT_ROOT)}"
            )

            if (
                not overwrite
                and is_valid_normalized_wav(destination)
            ):
                print("  SKIP: ya está normalizado")
                skipped += 1

            else:
                convert_with_ffmpeg(
                    source,
                    destination
                )

                if not is_valid_normalized_wav(destination):
                    raise RuntimeError(
                        "El WAV generado no cumple "
                        "mono / 16 kHz / PCM16."
                    )

                converted += 1
                print("  OK")

            new_row = dict(row)
            new_row["original_path"] = row["path"]
            new_row["path"] = relative_to_root(destination)

            normalized_rows.append(new_row)

        except Exception as e:
            print(
                f"  ERROR: {type(e).__name__}: {e}"
            )

            error_row = {
                "path": row.get("path", ""),
                "error_type": type(e).__name__,
                "error": str(e),
            }

            errors.append(error_row)

    return normalized_rows, errors, converted, skipped


# ============================================================
# GUARDADO
# ============================================================

def write_csv(path: Path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Mantener orden de columnas del primer registro
    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normaliza todos los WAV del dataset a "
            "mono, 16 kHz y PCM16 sin tocar los originales."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Vuelve a generar también los WAV ya normalizados."
        ),
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "No encuentro ffmpeg en PATH. "
            "Comprueba con: ffmpeg -version"
        )

    metadata_path = choose_metadata()

    print("==============================================")
    print("NORMALIZACIÓN DEL DATASET")
    print("==============================================")
    print(f"Metadata entrada : {metadata_path}")
    print(f"Salida           : {OUTPUT_ROOT}")
    print()
    print("Formato objetivo:")
    print("  Canales        : 1 (mono)")
    print("  Sample rate    : 16000 Hz")
    print("  Codec          : PCM16")
    print()
    print(
        "No se aplica normalización de volumen, "
        "denoise, trim de silencios ni compresión."
    )
    print()

    rows = load_rows(metadata_path)

    normalized_rows, errors, converted, skipped = normalize_dataset(
        rows,
        overwrite=args.overwrite,
    )

    write_csv(
        OUTPUT_METADATA,
        normalized_rows
    )

    if errors:
        write_csv(
            ERRORS_CSV,
            errors
        )
    elif ERRORS_CSV.exists():
        ERRORS_CSV.unlink()

    print()
    print("==============================================")
    print("RESUMEN")
    print("==============================================")
    print(f"Metadata original : {len(rows)}")
    print(f"Correctos         : {len(normalized_rows)}")
    print(f"Convertidos       : {converted}")
    print(f"Ya normalizados   : {skipped}")
    print(f"Errores           : {len(errors)}")
    print()
    print(f"Metadata nueva:")
    print(f"  {OUTPUT_METADATA}")

    if errors:
        print()
        print(f"Errores:")
        print(f"  {ERRORS_CSV}")

    print()
    print(
        "Los archivos originales de data\\ NO se han modificado."
    )


if __name__ == "__main__":
    main()
