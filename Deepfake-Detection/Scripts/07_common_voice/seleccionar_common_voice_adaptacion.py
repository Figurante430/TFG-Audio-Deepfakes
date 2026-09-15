from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT_DEFAULT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
SEED = 7301

N_SPEAKERS = 10
CLIPS_PER_SPEAKER = 20
MIN_DURATION = 1.5
MAX_DURATION = 12.0


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
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


def infer_excluded_client_ids(validated_rows, existing_metadata: Path | None):
    if existing_metadata is None or not existing_metadata.exists():
        return set()

    existing = read_csv(existing_metadata)
    excluded = {
        r.get("client_id", "").strip()
        for r in existing
        if r.get("client_id", "").strip()
    }

    if excluded:
        return excluded

    # Intento alternativo: mapear nombres originales si el metadata los conserva.
    basename_to_client = {}
    for r in validated_rows:
        p = r.get("path", "").strip()
        cid = r.get("client_id", "").strip()
        if p and cid:
            basename_to_client[Path(p).name] = cid

    candidate_cols = (
        "original_path",
        "source_path",
        "original_file",
        "source_file",
    )

    for r in existing:
        for col in candidate_cols:
            value = r.get(col, "").strip()
            if not value:
                continue
            cid = basename_to_client.get(Path(value).name)
            if cid:
                excluded.add(cid)

    return excluded


def duration_from_row(row):
    for key in ("duration_ms", "duration"):
        raw = row.get(key, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue

        if key == "duration_ms":
            return value / 1000.0
        return value

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Selecciona hablantes NUEVOS de Common Voice para adaptación de dominio."
    )
    parser.add_argument("--validated", type=Path, required=True,
                        help="Ruta a validated.tsv de Common Voice.")
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="Carpeta clips/ de Common Voice.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--existing-metadata", type=Path,
                        default=ROOT_DEFAULT / "external_test" / "metadata_external_real.csv")
    parser.add_argument("--n-speakers", type=int, default=N_SPEAKERS)
    parser.add_argument("--clips-per-speaker", type=int, default=CLIPS_PER_SPEAKER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-client-id", action="append", default=[],
                        help="Puede repetirse para excluir manualmente client_id.")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Necesitas ffmpeg y ffprobe disponibles en PATH.")

    rows = read_tsv(args.validated)

    if not rows:
        raise RuntimeError("validated.tsv está vacío.")

    if "client_id" not in rows[0] or "path" not in rows[0]:
        raise RuntimeError("validated.tsv debe contener columnas client_id y path.")

    excluded = infer_excluded_client_ids(rows, args.existing_metadata)
    excluded.update(x.strip() for x in args.exclude_client_id if x.strip())

    print(f"client_id excluidos automáticamente/manual: {len(excluded)}")

    # Seguridad: queremos separar los 5 speakers ya usados como diagnóstico.
    if args.existing_metadata.exists() and len(excluded) < 5:
        raise RuntimeError(
            "No he podido identificar con seguridad los 5 client_id del conjunto "
            "diagnóstico existente.\n"
            "Comprueba si external_test\\metadata_external_real.csv contiene una "
            "columna client_id u original_path/source_path.\n"
            "Si no, añade los IDs con --exclude-client-id ID (uno por speaker)."
        )

    by_client = defaultdict(list)

    for row in rows:
        cid = row.get("client_id", "").strip()
        rel = row.get("path", "").strip()

        if not cid or not rel or cid in excluded:
            continue

        d = duration_from_row(row)
        if d is not None and not (MIN_DURATION <= d <= MAX_DURATION):
            continue

        by_client[cid].append(row)

    # Pedimos margen por si algún clip falla o dura fuera de rango.
    min_candidates = args.clips_per_speaker + 5
    eligible = [
        cid for cid, clips in by_client.items()
        if len(clips) >= min_candidates
    ]

    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    out_root = args.root / "external_adaptation" / "real"
    metadata_path = args.root / "external_adaptation" / "metadata_external_adapt_train.csv"
    ids_path = args.root / "external_adaptation" / "selected_client_ids.txt"

    out_root.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    selected_ids = []

    speaker_index = 1

    for cid in eligible:
        if len(selected_ids) >= args.n_speakers:
            break

        candidates = list(by_client[cid])
        rng.shuffle(candidates)

        speaker_id = f"adapt_ext_{speaker_index:02d}"
        speaker_dir = out_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        accepted = []

        for row in candidates:
            src = args.clips_dir / row["path"]

            if not src.exists():
                continue

            d = duration_from_row(row)
            if d is None:
                try:
                    d = ffprobe_duration(src)
                except Exception:
                    continue

            if not (MIN_DURATION <= d <= MAX_DURATION):
                continue

            out_name = f"{speaker_id}_{len(accepted)+1:03d}.wav"
            dst = speaker_dir / out_name

            try:
                convert_to_wav(src, dst)
            except Exception:
                if dst.exists():
                    dst.unlink()
                continue

            accepted.append((row, src, dst, d))

            if len(accepted) >= args.clips_per_speaker:
                break

        if len(accepted) < args.clips_per_speaker:
            # No usamos parcialmente un speaker.
            for _, _, dst, _ in accepted:
                if dst.exists():
                    dst.unlink()
            try:
                speaker_dir.rmdir()
            except OSError:
                pass
            continue

        selected_ids.append(cid)

        for clip_idx, (row, src, dst, duration) in enumerate(accepted, 1):
            selected_rows.append({
                "path": str(dst.resolve()),
                "label": 0,
                "category": "real_external_adaptation",
                "generator": "real",
                "speaker_id": speaker_id,
                "source_speaker_id": speaker_id,
                "target_speaker_id": "",
                "utterance_id": f"{speaker_id}_{clip_idx:03d}",
                "source_generator": "common_voice",
                "target_model": "",
                "split": "train",
                "original_path": str(src.resolve()),
                "client_id": cid,
                "sentence": row.get("sentence", ""),
                "duration_seconds": f"{duration:.4f}",
            })

        print(f"[OK] {speaker_id}: {args.clips_per_speaker} clips")
        speaker_index += 1

    if len(selected_ids) < args.n_speakers:
        raise RuntimeError(
            f"Solo se pudieron seleccionar {len(selected_ids)} hablantes de "
            f"{args.n_speakers}."
        )

    fields = list(selected_rows[0].keys())
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    ids_path.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print("ADAPTACIÓN COMMON VOICE PREPARADA")
    print("=" * 72)
    print(f"Hablantes nuevos : {len(selected_ids)}")
    print(f"Clips por hablante: {args.clips_per_speaker}")
    print(f"Total reales      : {len(selected_rows)}")
    print(f"Metadata          : {metadata_path}")
    print(f"Client IDs        : {ids_path}")
    print()
    print("IMPORTANTE: estos audios son SOLO para TRAIN.")
    print("Los 100 Common Voice diagnósticos NO se añaden al entrenamiento.")


if __name__ == "__main__":
    main()
