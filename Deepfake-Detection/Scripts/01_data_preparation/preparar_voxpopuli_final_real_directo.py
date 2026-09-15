from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT_DEFAULT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

# Revisión histórica de VoxPopuli que contiene el split test español
# como TSV + un único tar.gz, sin necesidad de Parquet/pyarrow.
REVISION = "4ca6fb6d6bb37af66991d05974e6abcadd4b181e"
BASE = (
    "https://huggingface.co/datasets/facebook/voxpopuli/resolve/"
    + REVISION
)

TSV_URL = BASE + "/data/es/asr_test.tsv?download=true"
TAR_URL = BASE + "/data/es/test/test_part_0.tar.gz?download=true"

N_SPEAKERS = 10
CLIPS_PER_SPEAKER = 10
MIN_DURATION = 1.5
MAX_DURATION = 12.0

# Congelado antes de evaluar el detector.
FREEZE_TAG = "tfg_final_real_voxpopuli_es_legacy_v1_2026-09-15"

# Tomamos margen para duración/archivos inválidos.
MAX_SPEAKER_CANDIDATES = 40
MAX_CLIPS_PER_CANDIDATE_SPEAKER = 35

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "TFG-Deepfake-Detection/1.0"
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(
        f"{FREEZE_TAG}|{text}".encode("utf-8")
    ).hexdigest()


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está disponible en PATH.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe no está disponible en PATH.")


def download_with_resume(url: str, dst: Path, retries: int = 6):
    """
    Descarga con reanudación HTTP Range cuando el servidor lo permite.
    Solo stdlib; no requests/huggingface_hub/pyarrow.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing = dst.stat().st_size if dst.exists() else 0

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", None)
                content_range = resp.headers.get("Content-Range")
                content_length = resp.headers.get("Content-Length")

                # Si pedimos Range pero el servidor responde 200 completo,
                # reiniciamos para no concatenar un archivo duplicado.
                if existing > 0 and status == 200 and not content_range:
                    print("  El servidor no aceptó reanudación; reinicio descarga.")
                    existing = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing > 0 else "wb"

                total = None
                if content_range and "/" in content_range:
                    try:
                        total = int(content_range.rsplit("/", 1)[1])
                    except Exception:
                        total = None
                elif content_length:
                    try:
                        total = existing + int(content_length)
                    except Exception:
                        total = None

                downloaded = existing
                last_print = time.time()

                with dst.open(mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_print >= 1.0:
                            if total:
                                pct = 100.0 * downloaded / total
                                print(
                                    f"\r  {downloaded/1024**2:8.1f} MiB "
                                    f"/ {total/1024**2:8.1f} MiB "
                                    f"({pct:5.1f}%)",
                                    end="",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r  {downloaded/1024**2:8.1f} MiB",
                                    end="",
                                    flush=True,
                                )
                            last_print = now

                print()
                return

        except Exception as e:
            print(f"\n  intento {attempt}/{retries} falló: {e}")
            if attempt >= retries:
                raise
            wait = min(2 ** attempt, 20)
            print(f"  reintento en {wait}s...")
            time.sleep(wait)


def detect_column(fieldnames, candidates, human_name):
    lower_map = {
        str(name).strip().lower(): name
        for name in fieldnames
        if name is not None
    }

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise RuntimeError(
        f"No encuentro la columna de {human_name}.\n"
        f"Cabeceras TSV: {fieldnames}"
    )


def load_test_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t|,")
        except Exception:
            dialect = csv.excel_tab

        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    if not rows or not reader.fieldnames:
        raise RuntimeError("El TSV no contiene filas utilizables.")

    audio_col = detect_column(
        reader.fieldnames,
        ["audio_id", "id", "id_", "segment_id", "utt_id"],
        "audio_id",
    )
    speaker_col = detect_column(
        reader.fieldnames,
        ["speaker_id", "speaker", "speakerid"],
        "speaker_id",
    )

    def optional_col(options):
        lower = {x.lower(): x for x in reader.fieldnames if x}
        for o in options:
            if o.lower() in lower:
                return lower[o.lower()]
        return None

    raw_col = optional_col(["raw_text", "text", "transcript"])
    norm_col = optional_col(["normalized_text", "normalized"])
    gender_col = optional_col(["gender"])
    gold_col = optional_col(["is_gold_transcript", "gold"])
    accent_col = optional_col(["accent"])

    parsed = []

    for r in rows:
        audio_id = str(r.get(audio_col, "") or "").strip()
        speaker_id = str(r.get(speaker_col, "") or "").strip()

        if not audio_id or not speaker_id:
            continue

        parsed.append(
            {
                "audio_id": audio_id,
                "source_speaker_id": speaker_id,
                "raw_text": (r.get(raw_col, "") if raw_col else "") or "",
                "normalized_text": (
                    r.get(norm_col, "") if norm_col else ""
                ) or "",
                "gender": (r.get(gender_col, "") if gender_col else "") or "",
                "is_gold_transcript": (
                    r.get(gold_col, "") if gold_col else ""
                ) or "",
                "accent": (r.get(accent_col, "") if accent_col else "") or "",
            }
        )

    if not parsed:
        raise RuntimeError(
            "El TSV se leyó, pero no pude extraer audio_id/speaker_id."
        )

    return parsed, {
        "audio_col": audio_col,
        "speaker_col": speaker_col,
        "raw_col": raw_col,
        "norm_col": norm_col,
    }


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(
        cmd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    return float(out)


def convert_to_wav(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def member_audio_id(member_name: str) -> str:
    return Path(member_name).stem


def boolish(value):
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    return ""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara 100 reales de VoxPopuli ES test SIN pyarrow y "
            "SIN Dataset Viewer /rows. Descarga solo TSV + tar del test."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Borra y vuelve a descargar TSV/tar cacheados.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recrea el holdout. NO usar después de mirar resultados "
            "del detector."
        ),
    )
    args = parser.parse_args()

    ensure_ffmpeg()

    root = args.root
    out_root = root / "external_final_voxpopuli"
    cache_root = root / "_voxpopuli_test_cache"

    tsv_path = cache_root / "asr_test.tsv"
    tar_path = cache_root / "test_part_0.tar.gz"

    metadata_path = out_root / "metadata_voxpopuli_real.csv"
    manifest_path = out_root / "freeze_manifest.json"
    wav_root = out_root / "real"

    if manifest_path.exists() and not args.force:
        print("El holdout ya está congelado:")
        print(f"  {manifest_path}")
        print("No lo modifico.")
        return

    if args.force:
        if out_root.exists():
            shutil.rmtree(out_root, ignore_errors=True)

    if args.force_download:
        for p in (tsv_path, tar_path):
            if p.exists():
                p.unlink()

    cache_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("1) DESCARGA DEL METADATA TEST")
    print("=" * 72)
    print("TSV esperado: ~0.7 MB")
    if not tsv_path.exists():
        download_with_resume(TSV_URL, tsv_path)
    else:
        print(f"Ya existe: {tsv_path}")

    rows, detected = load_test_tsv(tsv_path)

    print(f"Filas válidas TSV: {len(rows)}")
    print(
        f"Columnas detectadas: audio={detected['audio_col']} | "
        f"speaker={detected['speaker_col']}"
    )

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["source_speaker_id"]].append(row)

    eligible = [
        spk for spk, clips in by_speaker.items()
        if len(clips) >= CLIPS_PER_SPEAKER
    ]
    eligible.sort(
        key=lambda spk: stable_hash(f"speaker|{spk}")
    )

    if len(eligible) < N_SPEAKERS:
        raise RuntimeError(
            f"Solo hay {len(eligible)} speakers con >= "
            f"{CLIPS_PER_SPEAKER} clips."
        )

    candidate_speakers = eligible[:MAX_SPEAKER_CANDIDATES]

    # Candidatos de clip deterministas por speaker.
    candidate_by_audio_id = {}
    ranked_by_speaker = {}

    for spk in candidate_speakers:
        clips = list(by_speaker[spk])
        clips.sort(
            key=lambda r: stable_hash(
                f"clip|{spk}|{r['audio_id']}"
            )
        )
        clips = clips[:MAX_CLIPS_PER_CANDIDATE_SPEAKER]
        ranked_by_speaker[spk] = clips

        for rank, row in enumerate(clips):
            item = dict(row)
            item["_rank"] = rank
            candidate_by_audio_id[item["audio_id"]] = item

    print(
        f"Speakers candidatos: {len(candidate_speakers)} | "
        f"clips candidatos: {len(candidate_by_audio_id)}"
    )

    print()
    print("=" * 72)
    print("2) DESCARGA DEL ARCHIVO TEST")
    print("=" * 72)
    print(
        "Este es el único archivo grande: ~628.6 MB. "
        "Se guarda en caché y admite reanudación."
    )

    if not tar_path.exists():
        download_with_resume(TAR_URL, tar_path)
    else:
        print(f"Ya existe: {tar_path}")

    print()
    print("=" * 72)
    print("3) EXTRACCIÓN DE CANDIDATOS")
    print("=" * 72)

    valid_by_speaker = defaultdict(list)
    matched = 0
    sample_member_names = []

    with tempfile.TemporaryDirectory(
        prefix="voxpopuli_extract_"
    ) as tmp:
        tmp_dir = Path(tmp)

        with tarfile.open(tar_path, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                if len(sample_member_names) < 5:
                    sample_member_names.append(member.name)

                audio_id = member_audio_id(member.name)

                row = candidate_by_audio_id.get(audio_id)
                if row is None:
                    continue

                matched += 1

                src_f = tar.extractfile(member)
                if src_f is None:
                    continue

                suffix = Path(member.name).suffix or ".audio"
                tmp_src = tmp_dir / f"candidate_{matched:05d}{suffix}"

                with tmp_src.open("wb") as out:
                    shutil.copyfileobj(src_f, out)

                try:
                    duration = ffprobe_duration(tmp_src)
                except Exception:
                    tmp_src.unlink(missing_ok=True)
                    continue

                if not (MIN_DURATION <= duration <= MAX_DURATION):
                    tmp_src.unlink(missing_ok=True)
                    continue

                # Convertimos a un WAV temporal con nombre seguro.
                tmp_wav = (
                    cache_root
                    / "candidate_wavs"
                    / row["source_speaker_id"]
                    / f"{stable_hash(audio_id)[:20]}.wav"
                )
                tmp_wav.parent.mkdir(parents=True, exist_ok=True)

                try:
                    convert_to_wav(tmp_src, tmp_wav)
                except Exception:
                    tmp_src.unlink(missing_ok=True)
                    continue

                tmp_src.unlink(missing_ok=True)

                valid_item = dict(row)
                valid_item["duration_seconds"] = duration
                valid_item["_candidate_wav"] = str(tmp_wav)
                valid_by_speaker[row["source_speaker_id"]].append(valid_item)

    if matched == 0:
        print("Ejemplos de nombres dentro del tar:")
        for x in sample_member_names:
            print(f"  {x}")
        print("Ejemplos de audio_id del TSV:")
        for x in list(candidate_by_audio_id)[:5]:
            print(f"  {x}")
        raise RuntimeError(
            "No hubo coincidencias entre audio_id del TSV y los nombres "
            "del tar. Pásame esta salida y adapto el mapeo."
        )

    print(f"Clips candidatos encontrados dentro del tar: {matched}")

    # Ordenamos nuevamente por ranking original y seleccionamos
    # los primeros 10 válidos por speaker.
    selected_source_speakers = []
    selected_items = []

    for spk in candidate_speakers:
        valid = valid_by_speaker.get(spk, [])
        valid.sort(key=lambda r: r["_rank"])

        if len(valid) < CLIPS_PER_SPEAKER:
            continue

        selected_source_speakers.append(spk)
        selected_items.append(valid[:CLIPS_PER_SPEAKER])

        if len(selected_source_speakers) >= N_SPEAKERS:
            break

    if len(selected_source_speakers) < N_SPEAKERS:
        counts = sorted(
            (
                (spk, len(valid_by_speaker.get(spk, [])))
                for spk in candidate_speakers
            ),
            key=lambda x: -x[1],
        )[:20]

        print("Mejores speakers por clips válidos:")
        for spk, n in counts:
            print(f"  {spk}: {n}")

        raise RuntimeError(
            f"Solo conseguí {len(selected_source_speakers)}/"
            f"{N_SPEAKERS} speakers con {CLIPS_PER_SPEAKER} "
            "clips válidos. Aumenta los márgenes del script."
        )

    print()
    print("=" * 72)
    print("4) CONGELANDO HOLDOUT FINAL")
    print("=" * 72)

    metadata_rows = []

    for speaker_num, (source_spk, clips) in enumerate(
        zip(selected_source_speakers, selected_items),
        1,
    ):
        public_spk = f"vox_ext_{speaker_num:02d}"
        speaker_dir = wav_root / public_spk
        speaker_dir.mkdir(parents=True, exist_ok=True)

        for clip_num, clip in enumerate(clips, 1):
            dst = speaker_dir / f"{public_spk}_{clip_num:03d}.wav"
            src = Path(clip["_candidate_wav"])
            shutil.copy2(src, dst)

            metadata_rows.append(
                {
                    "path": str(dst.resolve()),
                    "speaker_id": public_spk,
                    "source_speaker_id": source_spk,
                    "audio_id": clip["audio_id"],
                    "raw_text": clip["raw_text"],
                    "normalized_text": clip["normalized_text"],
                    "gender": clip["gender"],
                    "accent": clip["accent"],
                    "is_gold_transcript": boolish(
                        clip["is_gold_transcript"]
                    ),
                    "duration_seconds": f"{clip['duration_seconds']:.4f}",
                    "label": 0,
                    "category": "real_external_final",
                    "generator": "real",
                    "dataset": "voxpopuli_es_test_legacy_archive",
                    "freeze_tag": FREEZE_TAG,
                }
            )

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(metadata_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    manifest = {
        "freeze_tag": FREEZE_TAG,
        "dataset": "facebook/voxpopuli",
        "language": "es",
        "split": "test",
        "revision": REVISION,
        "source_tsv_url": TSV_URL,
        "source_tar_url": TAR_URL,
        "pyarrow_used": False,
        "pandas_used": False,
        "datasets_library_used": False,
        "dataset_viewer_rows_api_used": False,
        "selection": (
            "Deterministic SHA256 speaker/clip ranking, followed only "
            "by predeclared duration criterion 1.5-12 s."
        ),
        "n_speakers": N_SPEAKERS,
        "clips_per_speaker": CLIPS_PER_SPEAKER,
        "total_clips": len(metadata_rows),
        "selected_source_speaker_ids": selected_source_speakers,
        "metadata": str(metadata_path.resolve()),
        "warning": (
            "FINAL REAL HOLDOUT. Do not tune architecture, training data, "
            "augmentations or thresholds from detector results on this set."
        ),
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("VOXPOPULI FINAL REAL HOLDOUT — CONGELADO")
    print("=" * 72)
    print(f"Speakers : {len(selected_source_speakers)}")
    print(f"Audios   : {len(metadata_rows)}")
    print(f"WAVs     : {wav_root}")
    print(f"Metadata : {metadata_path}")
    print(f"Manifest : {manifest_path}")
    print()
    print(
        "Ya puedes borrar _voxpopuli_test_cache/candidate_wavs si quieres, "
        "pero conserva el tar para reproducibilidad."
    )
    print(
        "NO uses --force una vez que evalúes los detectores."
    )


if __name__ == "__main__":
    main()
