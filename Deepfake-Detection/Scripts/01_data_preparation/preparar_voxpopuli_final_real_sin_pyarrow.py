from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT_DEFAULT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

DATASET = "facebook/voxpopuli"
CONFIG = "es"
SPLIT = "test"

N_SPEAKERS = 10
CLIPS_PER_SPEAKER = 10
MIN_DURATION = 1.5
MAX_DURATION = 12.0

# Fijado ANTES de evaluar el detector.
FREEZE_TAG = "tfg_final_real_voxpopuli_es_v2_no_pyarrow_2026-09-15"

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "TFG-Deepfake-Detection/1.0"
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(
        f"{FREEZE_TAG}|{text}".encode("utf-8")
    ).hexdigest()


def http_json(url: str, retries: int = 5) -> dict:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )

            with urllib.request.urlopen(
                req,
                timeout=90,
            ) as resp:
                return json.loads(
                    resp.read().decode("utf-8")
                )

        except Exception as e:
            last_error = e

            if attempt < retries:
                wait = min(2 ** attempt, 15)
                print(
                    f"  HTTP/API intento {attempt}/{retries} falló: {e}"
                )
                print(f"  Reintento en {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"No se pudo consultar:\n{url}\nÚltimo error: {last_error}"
    )


def rows_url(offset: int, length: int) -> str:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_API}?{params}"


def fetch_all_rows():
    print(
        "Consultando metadata mediante la API REST del Dataset Viewer..."
    )
    print(
        "No se usa pyarrow, pandas, datasets ni huggingface_hub."
    )
    print()

    all_rows = []
    offset = 0
    total = None

    while total is None or offset < total:
        data = http_json(
            rows_url(
                offset=offset,
                length=PAGE_SIZE,
            )
        )

        page = data.get("rows", [])

        if total is None:
            total = int(
                data.get(
                    "num_rows_total",
                    len(page),
                )
            )
            print(
                f"Filas declaradas en {CONFIG}/{SPLIT}: {total}"
            )

        if not page:
            break

        for item in page:
            row_idx = item.get("row_idx")
            row = item.get("row", {})

            if row_idx is None:
                row_idx = offset + len(all_rows)

            all_rows.append(
                {
                    "row_idx": int(row_idx),
                    "row": row,
                }
            )

        offset += len(page)

        print(
            f"  metadata: {min(offset, total)}/{total}"
        )

        if len(page) < PAGE_SIZE:
            break

    if not all_rows:
        raise RuntimeError(
            "La API no devolvió filas. "
            "Puede ser una incidencia temporal del Dataset Viewer."
        )

    return all_rows


def extract_audio_url(row: dict) -> str | None:
    audio = row.get("audio")

    if isinstance(audio, str):
        if audio.startswith(
            ("http://", "https://")
        ):
            return audio
        return None

    if isinstance(audio, dict):
        for key in (
            "src",
            "url",
            "download_url",
        ):
            value = audio.get(key)

            if (
                isinstance(value, str)
                and value.startswith(
                    ("http://", "https://")
                )
            ):
                return value

    return None


def refresh_audio_url(row_idx: int) -> str:
    data = http_json(
        rows_url(
            offset=row_idx,
            length=1,
        )
    )

    rows = data.get("rows", [])

    if not rows:
        raise RuntimeError(
            f"No se pudo refrescar row_idx={row_idx}"
        )

    row = rows[0].get("row", {})
    url = extract_audio_url(row)

    if not url:
        raise RuntimeError(
            f"La API no devolvió URL de audio para row_idx={row_idx}.\n"
            f"Campo audio recibido: {row.get('audio')!r}"
        )

    return url


def download_file(
    url: str,
    dst: Path,
    retries: int = 3,
):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*",
                },
            )

            with urllib.request.urlopen(
                req,
                timeout=120,
            ) as resp, dst.open("wb") as f:
                shutil.copyfileobj(
                    resp,
                    f,
                    length=1024 * 1024,
                )

            if dst.stat().st_size <= 0:
                raise RuntimeError(
                    "archivo descargado vacío"
                )

            return

        except Exception as e:
            last_error = e

            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass

            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Error descargando audio: {last_error}"
    )


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()

    return float(out)


def convert_to_wav(
    src: Path,
    dst: Path,
):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]

    subprocess.run(
        cmd,
        check=True,
    )


def cleanup_dir(path: Path):
    if path.exists():
        shutil.rmtree(
            path,
            ignore_errors=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara un holdout REAL final de VoxPopuli ES "
            "sin pyarrow, pandas ni datasets. "
            "Usa únicamente la API REST de Hugging Face + ffmpeg."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recrear el holdout. NO usar después de mirar "
            "resultados del detector."
        ),
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg no está disponible en PATH."
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe no está disponible en PATH."
        )

    out_root = (
        args.root
        / "external_final_voxpopuli"
    )

    wav_root = out_root / "real"

    metadata_path = (
        out_root
        / "metadata_voxpopuli_real.csv"
    )

    manifest_path = (
        out_root
        / "freeze_manifest.json"
    )

    if (
        manifest_path.exists()
        and not args.force
    ):
        print(
            "El holdout ya está congelado:"
        )
        print(f"  {manifest_path}")
        print()
        print(
            "No lo modifico. Esto evita cambiar el test "
            "después de haber visto resultados."
        )
        return

    if args.force:
        cleanup_dir(out_root)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    wav_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Obtener metadata sin Parquet local
    # --------------------------------------------------------
    items = fetch_all_rows()

    by_speaker = defaultdict(list)

    missing_audio_url = 0

    for item in items:
        row = item["row"]

        speaker = str(
            row.get("speaker_id")
            or ""
        ).strip()

        audio_id = str(
            row.get("audio_id")
            or ""
        ).strip()

        if not speaker or not audio_id:
            continue

        audio_url = extract_audio_url(
            row
        )

        if not audio_url:
            missing_audio_url += 1

        by_speaker[speaker].append(
            {
                "row_idx": item["row_idx"],
                "source_speaker_id": speaker,
                "audio_id": audio_id,
                "audio_url": audio_url,
                "raw_text": row.get(
                    "raw_text",
                    "",
                )
                or "",
                "normalized_text": row.get(
                    "normalized_text",
                    "",
                )
                or "",
                "gender": row.get(
                    "gender",
                    "",
                )
                or "",
                "accent": row.get(
                    "accent",
                    "",
                )
                or "",
                "is_gold_transcript": bool(
                    row.get(
                        "is_gold_transcript",
                        False,
                    )
                ),
            }
        )

    eligible = [
        speaker
        for speaker, clips
        in by_speaker.items()
        if len(clips) >= CLIPS_PER_SPEAKER
    ]

    eligible.sort(
        key=lambda s: stable_hash(
            f"speaker|{s}"
        )
    )

    print()
    print(
        f"Speakers totales: {len(by_speaker)}"
    )
    print(
        f"Speakers con >= {CLIPS_PER_SPEAKER} clips: "
        f"{len(eligible)}"
    )

    if missing_audio_url:
        print(
            f"Aviso: {missing_audio_url} filas llegaron sin URL "
            "de audio; se refrescarán si fueran seleccionadas."
        )

    if len(eligible) < N_SPEAKERS:
        raise RuntimeError(
            f"No hay al menos {N_SPEAKERS} speakers "
            f"con {CLIPS_PER_SPEAKER} clips."
        )

    # --------------------------------------------------------
    # 2. Selección determinista + validación de duración
    # --------------------------------------------------------
    metadata_rows = []
    selected_source_speakers = []

    with tempfile.TemporaryDirectory(
        prefix="voxpopuli_tfg_"
    ) as tmp:
        tmp_dir = Path(tmp)

        for source_speaker in eligible:
            if (
                len(selected_source_speakers)
                >= N_SPEAKERS
            ):
                break

            speaker_number = (
                len(selected_source_speakers)
                + 1
            )

            public_speaker_id = (
                f"vox_ext_{speaker_number:02d}"
            )

            candidates = list(
                by_speaker[source_speaker]
            )

            candidates.sort(
                key=lambda c: stable_hash(
                    "clip|"
                    f"{source_speaker}|"
                    f"{c['audio_id']}"
                )
            )

            accepted = []

            print()
            print(
                f"Probando speaker candidato "
                f"{source_speaker} -> "
                f"{public_speaker_id}"
            )

            for candidate in candidates:
                if (
                    len(accepted)
                    >= CLIPS_PER_SPEAKER
                ):
                    break

                row_idx = int(
                    candidate["row_idx"]
                )

                url = candidate[
                    "audio_url"
                ]

                if not url:
                    try:
                        url = refresh_audio_url(
                            row_idx
                        )
                    except Exception as e:
                        print(
                            f"  skip row {row_idx}: "
                            f"sin URL ({e})"
                        )
                        continue

                src_tmp = (
                    tmp_dir
                    / f"row_{row_idx}.audio"
                )

                try:
                    download_file(
                        url,
                        src_tmp,
                    )
                except Exception:
                    # Las URLs de assets son temporales.
                    # Refrescamos la fila y probamos una vez más.
                    try:
                        url = refresh_audio_url(
                            row_idx
                        )

                        download_file(
                            url,
                            src_tmp,
                        )
                    except Exception as e:
                        print(
                            f"  skip {candidate['audio_id']}: "
                            f"descarga fallida ({e})"
                        )
                        continue

                try:
                    duration = ffprobe_duration(
                        src_tmp
                    )
                except Exception as e:
                    print(
                        f"  skip {candidate['audio_id']}: "
                        f"ffprobe ({e})"
                    )

                    try:
                        src_tmp.unlink()
                    except OSError:
                        pass

                    continue

                if not (
                    MIN_DURATION
                    <= duration
                    <= MAX_DURATION
                ):
                    try:
                        src_tmp.unlink()
                    except OSError:
                        pass

                    continue

                clip_number = (
                    len(accepted) + 1
                )

                speaker_dir = (
                    wav_root
                    / public_speaker_id
                )

                wav_name = (
                    f"{public_speaker_id}_"
                    f"{clip_number:03d}.wav"
                )

                wav_path = (
                    speaker_dir
                    / wav_name
                )

                try:
                    convert_to_wav(
                        src_tmp,
                        wav_path,
                    )
                except Exception as e:
                    print(
                        f"  skip {candidate['audio_id']}: "
                        f"ffmpeg ({e})"
                    )

                    try:
                        src_tmp.unlink()
                    except OSError:
                        pass

                    continue

                try:
                    src_tmp.unlink()
                except OSError:
                    pass

                accepted.append(
                    {
                        **candidate,
                        "path": str(
                            wav_path.resolve()
                        ),
                        "speaker_id": (
                            public_speaker_id
                        ),
                        "duration_seconds": (
                            duration
                        ),
                    }
                )

                print(
                    f"  OK {len(accepted):02d}/"
                    f"{CLIPS_PER_SPEAKER}: "
                    f"{candidate['audio_id']} "
                    f"({duration:.2f}s)"
                )

            if (
                len(accepted)
                < CLIPS_PER_SPEAKER
            ):
                print(
                    f"  DESCARTADO: solo "
                    f"{len(accepted)} clips válidos."
                )

                cleanup_dir(
                    wav_root
                    / public_speaker_id
                )

                continue

            selected_source_speakers.append(
                source_speaker
            )

            for clip in accepted:
                metadata_rows.append(
                    {
                        "path": clip["path"],
                        "speaker_id": (
                            clip["speaker_id"]
                        ),
                        "source_speaker_id": (
                            clip[
                                "source_speaker_id"
                            ]
                        ),
                        "audio_id": (
                            clip["audio_id"]
                        ),
                        "raw_text": (
                            clip["raw_text"]
                        ),
                        "normalized_text": (
                            clip[
                                "normalized_text"
                            ]
                        ),
                        "gender": (
                            clip["gender"]
                        ),
                        "accent": (
                            clip["accent"]
                        ),
                        "is_gold_transcript": int(
                            clip[
                                "is_gold_transcript"
                            ]
                        ),
                        "duration_seconds": (
                            f"{clip['duration_seconds']:.4f}"
                        ),
                        "label": 0,
                        "category": (
                            "real_external_final"
                        ),
                        "generator": "real",
                        "dataset": (
                            "voxpopuli_es_test"
                        ),
                        "source_row_idx": (
                            clip["row_idx"]
                        ),
                        "freeze_tag": FREEZE_TAG,
                    }
                )

    if (
        len(selected_source_speakers)
        != N_SPEAKERS
    ):
        raise RuntimeError(
            f"Solo se consiguieron "
            f"{len(selected_source_speakers)}/"
            f"{N_SPEAKERS} speakers válidos."
        )

    if (
        len(metadata_rows)
        != N_SPEAKERS
        * CLIPS_PER_SPEAKER
    ):
        raise RuntimeError(
            f"Se esperaban "
            f"{N_SPEAKERS * CLIPS_PER_SPEAKER} "
            f"audios y hay "
            f"{len(metadata_rows)}."
        )

    metadata_rows.sort(
        key=lambda r: (
            r["speaker_id"],
            r["path"],
        )
    )

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metadata_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    manifest = {
        "freeze_tag": FREEZE_TAG,
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "source": (
            "Hugging Face Dataset Viewer REST API"
        ),
        "local_parquet_used": False,
        "pyarrow_used": False,
        "pandas_used": False,
        "datasets_library_used": False,
        "selection": (
            "Deterministic SHA256 ranking of "
            "speaker_id and audio_id, followed only "
            "by the predeclared duration criterion. "
            "No detector prediction is used."
        ),
        "n_speakers": N_SPEAKERS,
        "clips_per_speaker": (
            CLIPS_PER_SPEAKER
        ),
        "total_clips": len(
            metadata_rows
        ),
        "duration_min_seconds": (
            MIN_DURATION
        ),
        "duration_max_seconds": (
            MAX_DURATION
        ),
        "audio_export": (
            "mono, 16000 Hz, PCM_16 WAV via ffmpeg"
        ),
        "selected_source_speaker_ids": (
            selected_source_speakers
        ),
        "metadata": str(
            metadata_path.resolve()
        ),
        "warning": (
            "FINAL REAL HOLDOUT. "
            "Do not tune model architecture, "
            "training data, augmentations or "
            "thresholds from its detector results."
        ),
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "VOXPOPULI FINAL REAL HOLDOUT — CONGELADO"
    )
    print("=" * 72)
    print(
        f"Speakers : "
        f"{len(selected_source_speakers)}"
    )
    print(
        f"Audios   : "
        f"{len(metadata_rows)}"
    )
    print(
        f"WAVs     : {wav_root}"
    )
    print(
        f"Metadata : {metadata_path}"
    )
    print(
        f"Manifest : {manifest_path}"
    )
    print()
    print(
        "No vuelvas a ejecutar con --force "
        "después de evaluar los modelos."
    )


if __name__ == "__main__":
    main()
