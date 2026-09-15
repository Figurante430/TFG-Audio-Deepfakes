from __future__ import annotations

import csv
import io
import random
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

ARCHIVE = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection\common_voice_es_26.tar.gz"
)

OUTPUT_ROOT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection\external_test\real"
)

N_SPEAKERS = 5
CLIPS_PER_SPEAKER = 20
SEED = 42

MIN_SECONDS = 1.5
MAX_SECONDS = 12.0

# Déjalo vacío para un test externo más diverso.
# Si quieres forzar acento peninsular, prueba por ejemplo:
# ACCENT_KEYWORDS = ["españa", "peninsular"]
ACCENT_KEYWORDS = []


def normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def find_validated_tsv(tar: tarfile.TarFile):
    candidates = []

    for member in tar.getmembers():
        name = member.name.replace("\\", "/").lower()

        if name.endswith("/validated.tsv") or name == "validated.tsv":
            candidates.append(member)

    if not candidates:
        raise RuntimeError(
            "No encuentro validated.tsv dentro del .tar.gz"
        )

    for member in candidates:
        normalized = member.name.replace("\\", "/").lower()
        if "/es/" in normalized:
            return member

    return candidates[0]


def load_metadata():
    print("[1/4] Buscando validated.tsv...")

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        member = find_validated_tsv(tar)

        print("TSV encontrado:")
        print(" ", member.name)

        f = tar.extractfile(member)

        if f is None:
            raise RuntimeError(
                "No puedo leer validated.tsv"
            )

        text = io.TextIOWrapper(
            f,
            encoding="utf-8",
            newline=""
        )

        reader = csv.DictReader(
            text,
            delimiter="\t"
        )

        rows = list(reader)

    print(
        f"Filas validated.tsv: {len(rows)}"
    )

    if not rows:
        raise RuntimeError(
            "validated.tsv está vacío"
        )

    required = {
        "client_id",
        "path",
        "sentence",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "Faltan columnas en validated.tsv: "
            + ", ".join(sorted(missing))
        )

    return rows


def accent_ok(row):
    if not ACCENT_KEYWORDS:
        return True

    values = []

    for key in (
        "accent",
        "accents",
        "variant",
    ):
        if key in row:
            values.append(
                normalize_text(row.get(key, ""))
            )

    combined = " ".join(values)

    return any(
        keyword.lower() in combined
        for keyword in ACCENT_KEYWORDS
    )


def choose_speakers(rows):
    print("[2/4] Seleccionando hablantes...")

    by_speaker = defaultdict(list)

    for row in rows:
        speaker = normalize_text(
            row.get("client_id", "")
        )

        clip_path = (
            row.get("path", "")
            or ""
        ).strip()

        if not speaker or not clip_path:
            continue

        if not accent_ok(row):
            continue

        by_speaker[speaker].append(row)

    eligible = [
        speaker
        for speaker, clips in by_speaker.items()
        if len(clips) >= CLIPS_PER_SPEAKER
    ]

    print(
        f"Hablantes con >= "
        f"{CLIPS_PER_SPEAKER} clips: "
        f"{len(eligible)}"
    )

    if len(eligible) < N_SPEAKERS:
        raise RuntimeError(
            f"Solo hay {len(eligible)} hablantes "
            f"elegibles; necesito {N_SPEAKERS}."
        )

    rng = random.Random(SEED)
    rng.shuffle(eligible)

    selected = eligible[:N_SPEAKERS]

    selection = {}

    for idx, speaker in enumerate(
        selected,
        1
    ):
        clips = list(
            by_speaker[speaker]
        )

        rng.shuffle(clips)

        selection[speaker] = {
            "external_id": (
                f"speaker_ext_{idx:02d}"
            ),
            "candidates": clips[
                :CLIPS_PER_SPEAKER * 3
            ],
        }

        print(
            f"  speaker_ext_{idx:02d}: "
            f"{len(by_speaker[speaker])} "
            "clips disponibles"
        )

    return selection


def find_clip_members(
    tar: tarfile.TarFile,
    wanted_basenames: set[str],
):
    result = {}

    for member in tar.getmembers():
        if not member.isfile():
            continue

        name = member.name.replace(
            "\\",
            "/"
        )

        basename = Path(name).name

        if basename in wanted_basenames:
            result[basename] = member

    return result


def duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
    ).strip()

    return float(out)


def convert_to_wav(
    source: Path,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def extract_selected(selection):
    print(
        "[3/4] Extrayendo y convirtiendo "
        "clips seleccionados..."
    )

    wanted = set()

    for _, info in selection.items():
        for row in info["candidates"]:
            wanted.add(
                Path(row["path"]).name
            )

    manifest_rows = []

    with tarfile.open(
        ARCHIVE,
        "r:gz"
    ) as tar:

        members = find_clip_members(
            tar,
            wanted,
        )

        print(
            f"Clips encontrados en tar: "
            f"{len(members)}/{len(wanted)}"
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            for speaker, info in selection.items():

                ext_id = info[
                    "external_id"
                ]

                out_dir = (
                    OUTPUT_ROOT / ext_id
                )

                out_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                accepted = 0

                for row in info[
                    "candidates"
                ]:
                    if accepted >= CLIPS_PER_SPEAKER:
                        break

                    basename = Path(
                        row["path"]
                    ).name

                    member = members.get(
                        basename
                    )

                    if member is None:
                        continue

                    src = tar.extractfile(
                        member
                    )

                    if src is None:
                        continue

                    temp_src = (
                        tmpdir / basename
                    )

                    with temp_src.open(
                        "wb"
                    ) as f:
                        shutil.copyfileobj(
                            src,
                            f
                        )

                    try:
                        dur = duration_seconds(
                            temp_src
                        )
                    except Exception:
                        continue

                    if not (
                        MIN_SECONDS
                        <= dur
                        <= MAX_SECONDS
                    ):
                        continue

                    accepted += 1

                    dst = (
                        out_dir
                        / f"{ext_id}_{accepted:03d}.wav"
                    )

                    convert_to_wav(
                        temp_src,
                        dst
                    )

                    manifest_rows.append({
                        "path": str(dst),
                        "speaker_id": ext_id,
                        "source_client_id": speaker,
                        "sentence": row.get(
                            "sentence",
                            ""
                        ),
                        "source_filename": basename,
                        "source_duration_seconds": round(
                            dur,
                            3
                        ),
                        "label": 0,
                        "category": "real_external",
                        "generator": "real",
                        "dataset": "common_voice_26_es",
                    })

                    print(
                        f"  {ext_id}: "
                        f"{accepted:02d}/"
                        f"{CLIPS_PER_SPEAKER}"
                    )

                if accepted < CLIPS_PER_SPEAKER:
                    raise RuntimeError(
                        f"{ext_id}: solo he podido "
                        f"obtener {accepted} clips válidos."
                    )

    return manifest_rows


def save_manifest(rows):
    print(
        "[4/4] Guardando metadata externa..."
    )

    manifest = (
        OUTPUT_ROOT.parent
        / "metadata_external_real.csv"
    )

    with manifest.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Audios reales externos: "
        f"{len(rows)}"
    )

    print(
        f"Metadata: {manifest}"
    )


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"No encuentro:\n{ARCHIVE}"
        )

    if shutil.which(
        "ffmpeg"
    ) is None:
        raise RuntimeError(
            "No encuentro ffmpeg en PATH."
        )

    if shutil.which(
        "ffprobe"
    ) is None:
        raise RuntimeError(
            "No encuentro ffprobe en PATH."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_metadata()
    selection = choose_speakers(rows)
    manifest_rows = extract_selected(
        selection
    )
    save_manifest(
        manifest_rows
    )


if __name__ == "__main__":
    main()
