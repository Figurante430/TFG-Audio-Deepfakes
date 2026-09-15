from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
DATA_DIR = ROOT / "data"

REAL_DIR = DATA_DIR / "real"
TTS_DIR = DATA_DIR / "tts"
STS_DIR = DATA_DIR / "sts"

OUTPUT_CSV = DATA_DIR / "metadata.csv"

REF_TO_SPEAKER = {
    1: "gonzalo",
    2: "marta",
    3: "noelia",
    4: "rober",
    5: "samu",
}

KNNVC_TARGET = {
    1: "marta",
    2: "noelia",
    3: "rober",
    4: "samu",
    5: "gonzalo",
}

RVC_TARGET_SPEAKER = "mario"
RVC_TARGET_MODEL = "Mario_40"


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_wavs(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.rglob("*.wav") if p.is_file()],
        key=natural_key,
    )


def relative_path(path: Path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def extract_ref_id(path: Path):
    match = re.search(r"(?:^|_)ref([1-5])(?:_|$)", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_utterance_id(path: Path, fallback: int):
    match = re.match(r"^(\d+)", path.stem)
    if match:
        return match.group(1).zfill(3)
    return str(fallback).zfill(3)


def add_row(
    rows,
    *,
    path,
    label,
    category,
    generator,
    speaker_id,
    source_speaker_id="",
    target_speaker_id="",
    utterance_id="",
    source_generator="",
    target_model="",
):
    rows.append({
        "path": relative_path(path),
        "label": label,
        "category": category,
        "generator": generator,
        "speaker_id": speaker_id,
        "source_speaker_id": source_speaker_id,
        "target_speaker_id": target_speaker_id,
        "utterance_id": utterance_id,
        "source_generator": source_generator,
        "target_model": target_model,
    })


def scan_real(rows):
    import wave

    print("\n================ REAL ================")

    if not REAL_DIR.exists():
        print(f"[WARN] No existe: {REAL_DIR}")
        return

    speaker_dirs = sorted(
        [p for p in REAL_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )

    for speaker_dir in speaker_dirs:

        speaker_id = speaker_dir.name.strip().lower()

        wavs = sorted(
            [
                p
                for p in speaker_dir.rglob("*.wav")
                if p.is_file()
            ],
            key=natural_key,
        )

        # ----------------------------------------------------
        # Obtener duración para poder identificar el WAV largo
        # original del que se sacaron las 40 frases
        # ----------------------------------------------------

        wavs_with_duration = []

        for wav in wavs:
            try:
                with wave.open(str(wav), "rb") as wf:
                    duration = (
                        wf.getnframes()
                        / wf.getframerate()
                    )
            except Exception:
                duration = 0

            wavs_with_duration.append(
                (wav, duration)
            )

        print(
            f"[REAL] {speaker_id}: "
            f"{len(wavs_with_duration)} WAV encontrados"
        )

        # ----------------------------------------------------
        # Para los cinco hablantes de las frases:
        # eliminar el audio largo original.
        #
        # Si hay 41 archivos, conservamos los 40 más cortos.
        # ----------------------------------------------------

        if (
            speaker_id in {
                "gonzalo",
                "marta",
                "noelia",
                "rober",
                "samu",
            }
            and len(wavs_with_duration) == 41
        ):

            # Ordenar temporalmente por duración
            by_duration = sorted(
                wavs_with_duration,
                key=lambda x: x[1]
            )

            # Los 40 más cortos = frases individuales
            selected = by_duration[:40]

            # El más largo = grabación original
            excluded = by_duration[-1]

            print(
                f"  Excluido original largo: "
                f"{excluded[0].name} "
                f"({excluded[1]:.1f}s)"
            )

            # Volver al orden natural para asignar 001..040
            selected = sorted(
                selected,
                key=lambda x: natural_key(x[0])
            )

        else:
            # Mario u otros hablantes:
            # conservar todos sus WAV
            selected = wavs_with_duration

        print(
            f"  -> {len(selected)} audios "
            f"incluidos en metadata"
        )

        # ----------------------------------------------------
        # Añadir metadata
        # ----------------------------------------------------

        for idx, (wav, duration) in enumerate(
            selected,
            1
        ):

            if speaker_id == "mario":
                utterance_id = (
                    f"mario_real_{idx:03d}"
                )
            else:
                utterance_id = str(idx).zfill(3)

            add_row(
                rows,
                path=wav,
                label=0,
                category="real",
                generator="real",
                speaker_id=speaker_id,
                source_speaker_id=speaker_id,
                target_speaker_id="",
                utterance_id=utterance_id,
                source_generator="real",
                target_model="",
            )


def scan_tts(rows):
    print("\n================ TTS ================")

    if not TTS_DIR.exists():
        print(f"[WARN] No existe: {TTS_DIR}")
        return

    model_dirs = sorted(
        [p for p in TTS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )

    for model_dir in model_dirs:
        generator = model_dir.name.lower()
        wavs = get_wavs(model_dir)

        print(f"[TTS] {generator}: {len(wavs)} WAV")

        for idx, wav in enumerate(wavs, 1):
            ref_id = extract_ref_id(wav)
            utterance_id = extract_utterance_id(wav, idx)

            if ref_id is None:
                speaker_id = "unknown"
                print(f"  [WARN] No puedo extraer ref1..ref5 de: {wav.name}")
            else:
                speaker_id = REF_TO_SPEAKER[ref_id]

            add_row(
                rows,
                path=wav,
                label=1,
                category="tts",
                generator=generator,
                speaker_id=speaker_id,
                source_speaker_id="",
                target_speaker_id=speaker_id,
                utterance_id=utterance_id,
                source_generator="text",
                target_model="",
            )


def scan_sts(rows):
    print("\n================ STS ================")

    if not STS_DIR.exists():
        print(f"[WARN] No existe: {STS_DIR}")
        return

    model_dirs = sorted(
        [
            p for p in STS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        ],
        key=lambda p: p.name.lower(),
    )

    for model_dir in model_dirs:
        generator = model_dir.name.lower()
        wavs = get_wavs(model_dir)

        print(f"[STS] {generator}: {len(wavs)} WAV")

        for idx, wav in enumerate(wavs, 1):
            ref_id = extract_ref_id(wav)
            utterance_id = extract_utterance_id(wav, idx)

            if ref_id is None:
                source_speaker = "unknown"
                print(f"  [WARN] No puedo extraer ref1..ref5 de: {wav.name}")
            else:
                source_speaker = REF_TO_SPEAKER[ref_id]

            if generator == "knnvc":
                target_speaker = (
                    KNNVC_TARGET[ref_id] if ref_id is not None else "unknown"
                )
                target_model = ""

            elif generator == "rvc":
                target_speaker = RVC_TARGET_SPEAKER
                target_model = RVC_TARGET_MODEL

            else:
                target_speaker = "unknown"
                target_model = ""
                print(
                    f"  [WARN] Modelo STS desconocido '{generator}'. "
                    "speaker_id queda como unknown."
                )

            add_row(
                rows,
                path=wav,
                label=1,
                category="sts",
                generator=generator,
                speaker_id=target_speaker,
                source_speaker_id=source_speaker,
                target_speaker_id=target_speaker,
                utterance_id=utterance_id,
                source_generator="confucius4",
                target_model=target_model,
            )


def validate(rows):
    print("\n================ VALIDACIÓN ================")

    paths = [row["path"] for row in rows]
    duplicates = [
        path for path, count in Counter(paths).items() if count > 1
    ]

    if duplicates:
        print(f"[WARN] Hay {len(duplicates)} rutas duplicadas.")
        for path in duplicates[:10]:
            print("  ", path)
    else:
        print("[OK] No hay rutas duplicadas.")

    unknown_speakers = sum(
        row["speaker_id"] == "unknown" for row in rows
    )

    if unknown_speakers:
        print(
            f"[WARN] Hay {unknown_speakers} muestras con speaker_id='unknown'."
        )
    else:
        print("[OK] Todos los speaker_id están identificados.")


def print_summary(rows):
    print("\n================ RESUMEN ================")

    total = len(rows)
    real = sum(row["label"] == 0 for row in rows)
    fake = sum(row["label"] == 1 for row in rows)

    print(f"TOTAL : {total}")
    print(f"REAL  : {real}")
    print(f"FAKE  : {fake}")

    print("\nPor categoría:")
    for name, count in sorted(
        Counter(row["category"] for row in rows).items()
    ):
        print(f"  {name:12s} {count}")

    print("\nPor generador:")
    for name, count in sorted(
        Counter(row["generator"] for row in rows).items()
    ):
        print(f"  {name:12s} {count}")

    print("\nPor speaker final:")
    for name, count in sorted(
        Counter(row["speaker_id"] for row in rows).items()
    ):
        print(f"  {name:12s} {count}")


def save_csv(rows):
    fieldnames = [
        "path",
        "label",
        "category",
        "generator",
        "speaker_id",
        "source_speaker_id",
        "target_speaker_id",
        "utterance_id",
        "source_generator",
        "target_model",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV guardado en:\n{OUTPUT_CSV}")


def main():
    rows = []

    scan_real(rows)
    scan_tts(rows)
    scan_sts(rows)

    validate(rows)
    print_summary(rows)
    save_csv(rows)


if __name__ == "__main__":
    main()
