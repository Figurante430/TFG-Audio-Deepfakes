from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

import csv
import hashlib
import json
import re
import sys

import numpy as np
import soundfile as sf


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection"
)

OUT_DIR = ROOT / "external_final_holdout"
OUT_CSV = OUT_DIR / "metadata_external_final.csv"
OUT_MANIFEST = OUT_DIR / "freeze_manifest_external_final.json"

EXPECTED_REAL = 100
EXPECTED_FAKE = 120
EXPECTED_TOTAL = 220


# =====================================================================
# DATASETS
#
# allow_global_hash_anchor=True:
#
# VoxPopuli fue congelado previamente, pero ese congelado antiguo no
# almacenó SHA-256 individual por WAV.
#
# Como TODAVÍA NO se han evaluado detectores, fijaremos los hashes
# actuales en este cierre global.
#
# Para todos los FAKE esto queda False: deben poder verificarse contra
# su hash histórico anterior.
# =====================================================================

DATASETS = [

    {
        "name": "VoxPopuli real",
        "class": "real",
        "label": 0,
        "generator": "VoxPopuli",
        "expected": 100,

        "wav_dir":
            ROOT
            / "external_final_voxpopuli"
            / "real",

        "metadata":
            ROOT
            / "external_final_voxpopuli"
            / "metadata_voxpopuli_real.csv",

        "manifest":
            ROOT
            / "external_final_voxpopuli"
            / "freeze_manifest.json",

        "allow_global_hash_anchor":
            True,
    },

    {
        "name": "RVC Mario_40",
        "class": "fake",
        "label": 1,
        "generator": "RVC",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "rvc",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_rvc.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_rvc.json",

        "allow_global_hash_anchor":
            False,
    },

    {
        "name": "Qwen3-TTS",
        "class": "fake",
        "label": 1,
        "generator": "Qwen3-TTS",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "qwen",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_qwen.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_qwen.json",

        "allow_global_hash_anchor":
            False,
    },

    {
        "name": "Fun-CosyVoice3",
        "class": "fake",
        "label": 1,
        "generator": "Fun-CosyVoice3",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "cosyvoice",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_cosyvoice.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_cosyvoice.json",

        "allow_global_hash_anchor":
            False,
    },

    {
        "name": "Confucius4-TTS",
        "class": "fake",
        "label": 1,
        "generator": "Confucius4-TTS",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "confucius",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_confucius.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_confucius.json",

        "allow_global_hash_anchor":
            False,
    },

    {
        "name": "OpenVoice V2",
        "class": "fake",
        "label": 1,
        "generator": "OpenVoice V2",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "openvoice",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_openvoice.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_openvoice.json",

        "allow_global_hash_anchor":
            False,
    },

    {
        "name": "VoxCPM2",
        "class": "fake",
        "label": 1,
        "generator": "VoxCPM2",
        "expected": 20,

        "wav_dir":
            ROOT
            / "external_final_fake"
            / "fake"
            / "voxcpm2",

        "metadata":
            ROOT
            / "external_final_fake"
            / "metadata_voxcpm2.csv",

        "manifest":
            ROOT
            / "external_final_fake"
            / "freeze_manifest_voxcpm2.json",

        "allow_global_hash_anchor":
            False,
    },
]


EXPECTED_GENERATOR_COUNTS = {

    "VoxPopuli": 100,
    "RVC": 20,
    "Qwen3-TTS": 20,
    "Fun-CosyVoice3": 20,
    "Confucius4-TTS": 20,
    "OpenVoice V2": 20,
    "VoxCPM2": 20,
}


# =====================================================================
# UTILS
# =====================================================================

def fail(message):

    print()
    print("=" * 72)
    print("ERROR")
    print("=" * 72)
    print(message)
    print("=" * 72)

    sys.exit(1)


def sha256(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def is_sha256(value):

    return (
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9a-fA-F]{64}",
            value.strip()
        )
        is not None
    )


def load_csv(path: Path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def audio_info(path: Path):

    try:

        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True
        )

    except Exception as exc:

        fail(
            f"No se puede leer:\n"
            f"{path}\n\n"
            f"{type(exc).__name__}: {exc}"
        )


    if audio.size == 0:

        fail(
            f"Audio vacío:\n{path}"
        )


    if not np.isfinite(audio).all():

        fail(
            f"NaN o Inf en:\n{path}"
        )


    frames = int(audio.shape[0])
    channels = int(audio.shape[1])
    sr = int(sr)

    return {

        "sample_rate":
            sr,

        "channels":
            channels,

        "frames":
            frames,

        "duration_s":
            float(frames / sr),
    }


# =====================================================================
# FIND WAV NAME
# =====================================================================

def find_filename(row):

    if not isinstance(row, dict):
        return None


    preferred = [

        "file",
        "filename",
        "wav",
        "wav_file",
        "audio_file",
        "output_file",
        "final_file",
        "relative_path",
        "path",
        "file_path",
        "wav_path",
        "audio_path",
        "source_file",
    ]


    for key in preferred:

        value = row.get(key)

        if (
            isinstance(value, str)
            and value.lower().endswith(".wav")
        ):

            return Path(value).name


    for value in row.values():

        if (
            isinstance(value, str)
            and value.lower().endswith(".wav")
        ):

            return Path(value).name


    return None


# =====================================================================
# FIND SHA256
# =====================================================================

def find_sha256(row):

    if not isinstance(row, dict):
        return None


    preferred = [

        "sha256",
        "wav_sha256",
        "file_sha256",
        "audio_sha256",
        "hash_sha256",
        "output_sha256",
    ]


    for key in preferred:

        value = row.get(key)

        if is_sha256(value):

            return (
                value
                .strip()
                .lower()
            )


    for key, value in row.items():

        if (
            "sha256"
            in str(key).lower()
            and is_sha256(value)
        ):

            return (
                value
                .strip()
                .lower()
            )


    return None


# =====================================================================
# EXTRACT RECORDS RECURSIVELY FROM MANIFEST
# =====================================================================

def extract_manifest_records(manifest):

    records = {}


    def save(
        filename,
        digest,
        extra=None
    ):

        if not filename:
            return

        if not is_sha256(digest):
            return


        filename = Path(
            str(filename)
        ).name

        digest = (
            str(digest)
            .strip()
            .lower()
        )


        record = {

            "file":
                filename,

            "sha256":
                digest,
        }


        if isinstance(extra, dict):

            record.update(extra)

            record["file"] = filename
            record["sha256"] = digest


        current = records.get(
            filename
        )


        if (
            current is None
            or len(record) > len(current)
        ):

            records[
                filename
            ] = record


    def walk(obj):

        if isinstance(obj, dict):

            # Example:
            # "audio.wav": "<sha256>"
            for key, value in obj.items():

                if (
                    isinstance(key, str)
                    and key.lower().endswith(".wav")
                    and is_sha256(value)
                ):

                    save(
                        key,
                        value
                    )


            filename = find_filename(
                obj
            )

            digest = find_sha256(
                obj
            )


            if filename and digest:

                save(
                    filename,
                    digest,
                    obj
                )


            for value in obj.values():

                walk(value)


        elif isinstance(obj, list):

            for value in obj:

                walk(value)


    walk(manifest)

    return records


# =====================================================================
# METADATA INDEX
# =====================================================================

def metadata_index(rows):

    result = {}


    for row in rows:

        filename = find_filename(
            row
        )

        if not filename:
            continue


        if filename in result:

            fail(
                "Nombre WAV duplicado "
                "en metadata:\n"
                f"{filename}"
            )


        result[
            filename
        ] = row


    return result


# =====================================================================
# SPEAKER
# =====================================================================

def find_speaker(
    filename,
    metadata_row,
    manifest_row
):

    keys = [

        "speaker",
        "speaker_id",
        "speaker_name",
        "source_speaker",
        "reference_speaker",
    ]


    for source in [
        metadata_row,
        manifest_row,
    ]:

        if not isinstance(
            source,
            dict
        ):
            continue


        for key in keys:

            value = source.get(
                key
            )

            if value not in [
                None,
                ""
            ]:

                return str(value)


    # VoxPopuli fallback
    match = re.search(
        r"(vox_ext_\d+)",
        filename,
        re.IGNORECASE
    )


    if match:

        return (
            match
            .group(1)
            .lower()
        )


    return ""


# =====================================================================
# OUTPUT SAFETY
# =====================================================================

if OUT_CSV.exists():

    fail(
        f"Ya existe:\n{OUT_CSV}\n\n"
        "No se sobrescribe."
    )


if OUT_MANIFEST.exists():

    fail(
        f"Ya existe:\n{OUT_MANIFEST}\n\n"
        "No se sobrescribe."
    )


if OUT_DIR.exists():

    contents = list(
        OUT_DIR.iterdir()
    )

    if contents:

        fail(
            f"La carpeta existe y no está vacía:\n"
            f"{OUT_DIR}"
        )


# =====================================================================
# GLOBAL BUILD
# =====================================================================

global_rows = []

source_manifests = []
source_metadata = []

warnings = []


for dataset in DATASETS:

    print()
    print("-" * 72)
    print(dataset["name"])
    print("-" * 72)


    # =================================================================
    # PATHS
    # =================================================================

    for path in [

        dataset["wav_dir"],
        dataset["metadata"],
        dataset["manifest"],

    ]:

        if not path.exists():

            fail(
                f"{dataset['name']}:\n"
                f"No existe:\n{path}"
            )


    # =================================================================
    # MANIFEST
    # =================================================================

    with dataset[
        "manifest"
    ].open(
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)


    status = manifest.get(
        "status"
    )


    if (
        status is not None
        and str(status).upper()
        != "FROZEN"
    ):

        fail(
            f"{dataset['name']}:\n"
            f"status={status}"
        )


    manifest_records = (
        extract_manifest_records(
            manifest
        )
    )


    print(
        "Manifest: "
        f"{len(manifest_records)} "
        "WAV con hash histórico"
    )


    # =================================================================
    # METADATA
    # =================================================================

    metadata_rows = load_csv(
        dataset["metadata"]
    )


    if (
        len(metadata_rows)
        != dataset["expected"]
    ):

        fail(
            f"{dataset['name']}:\n"
            f"metadata={len(metadata_rows)}, "
            f"esperaba {dataset['expected']}"
        )


    metadata_records = (
        metadata_index(
            metadata_rows
        )
    )


    print(
        f"Metadata: {len(metadata_rows)} filas, "
        f"{len(metadata_records)} asociadas a WAV"
    )


    # =================================================================
    # WAVS RECURSIVELY
    # =================================================================

    wavs = sorted(

        [

            path

            for path
            in dataset[
                "wav_dir"
            ].rglob("*")

            if (
                path.is_file()
                and path.suffix.lower()
                == ".wav"
            )

        ],

        key=lambda p:
            p
            .relative_to(
                dataset[
                    "wav_dir"
                ]
            )
            .as_posix()
    )


    if (
        len(wavs)
        != dataset["expected"]
    ):

        fail(
            f"{dataset['name']}:\n"
            f"Hay {len(wavs)} WAV; "
            f"esperaba {dataset['expected']}.\n\n"
            f"Ruta:\n"
            f"{dataset['wav_dir']}"
        )


    print(
        f"WAV encontrados: {len(wavs)}"
    )


    # =================================================================
    # BASENAME UNIQUENESS
    # =================================================================

    basenames = [
        wav.name
        for wav in wavs
    ]


    basename_counts = Counter(
        basenames
    )


    duplicated_names = [

        filename

        for filename, count
        in basename_counts.items()

        if count > 1
    ]


    if duplicated_names:

        fail(
            f"{dataset['name']}:\n"
            "Hay basenames duplicados:\n"
            + "\n".join(
                sorted(
                    duplicated_names
                )
            )
        )


    # =================================================================
    # VERIFY EACH FILE
    # =================================================================

    verified_old_hash = 0
    anchored_now = 0


    for wav in wavs:

        current_hash = (
            sha256(wav)
            .lower()
        )


        manifest_row = (
            manifest_records.get(
                wav.name,
                {}
            )
        )


        metadata_row = (
            metadata_records.get(
                wav.name,
                {}
            )
        )


        historic_hash = (
            find_sha256(
                manifest_row
            )
        )


        if not historic_hash:

            historic_hash = (
                find_sha256(
                    metadata_row
                )
            )


        # -------------------------------------------------------------
        # CASE A:
        # Historical hash exists -> verify.
        # -------------------------------------------------------------

        if historic_hash:

            if (
                current_hash
                != historic_hash.lower()
            ):

                fail(
                    "HASH MODIFICADO\n\n"
                    f"Dataset : {dataset['name']}\n"
                    f"WAV     : {wav}\n\n"
                    f"Histórico:\n"
                    f"{historic_hash}\n\n"
                    f"Actual:\n"
                    f"{current_hash}"
                )


            integrity_status = (
                "VERIFIED_AGAINST_PREVIOUS_FREEZE"
            )

            verified_old_hash += 1


        # -------------------------------------------------------------
        # CASE B:
        # No historical per-WAV hash.
        #
        # ONLY ALLOWED FOR VOXPOPULI REAL.
        # We create its immutable SHA-256 anchor NOW,
        # before detector evaluation.
        # -------------------------------------------------------------

        else:

            if not dataset[
                "allow_global_hash_anchor"
            ]:

                fail(
                    f"{dataset['name']} / "
                    f"{wav.name}\n\n"
                    "No existe SHA-256 histórico y "
                    "este dataset NO permite crear "
                    "un nuevo anclaje global."
                )


            integrity_status = (
                "ANCHORED_AT_GLOBAL_FREEZE"
            )

            anchored_now += 1


        # -------------------------------------------------------------
        # AUDIO QC
        # -------------------------------------------------------------

        info = audio_info(
            wav
        )


        if (
            info["duration_s"]
            < 0.1
        ):

            fail(
                f"Audio demasiado corto:\n"
                f"{wav}"
            )


        # -------------------------------------------------------------
        # SPEAKER
        # -------------------------------------------------------------

        speaker = find_speaker(

            wav.name,

            metadata_row,

            manifest_row
        )


        # -------------------------------------------------------------
        # GLOBAL RELATIVE PATH
        # -------------------------------------------------------------

        relative_path = (
            wav
            .relative_to(
                ROOT
            )
            .as_posix()
        )


        # -------------------------------------------------------------
        # GLOBAL ROW
        # -------------------------------------------------------------

        global_rows.append({

            "id":
                len(global_rows) + 1,

            "relative_path":
                relative_path,

            "file":
                wav.name,

            "label":
                dataset["label"],

            "class":
                dataset["class"],

            "generator":
                dataset["generator"],

            "speaker":
                speaker,

            "sample_rate":
                info[
                    "sample_rate"
                ],

            "channels":
                info[
                    "channels"
                ],

            "frames":
                info[
                    "frames"
                ],

            "duration_s":
                f"{info['duration_s']:.6f}",

            "bytes":
                wav.stat().st_size,

            "sha256":
                current_hash,

            "integrity_status":
                integrity_status,

            "source_metadata":
                dataset[
                    "metadata"
                ]
                .relative_to(
                    ROOT
                )
                .as_posix(),

            "source_manifest":
                dataset[
                    "manifest"
                ]
                .relative_to(
                    ROOT
                )
                .as_posix(),
        })


    # =================================================================
    # DATASET SUMMARY
    # =================================================================

    print(
        "Verificados contra congelado previo: "
        f"{verified_old_hash}"
    )

    print(
        "SHA-256 anclados en cierre global:    "
        f"{anchored_now}"
    )

    print(
        "QC                                  : OK"
    )


    # =================================================================
    # SOURCE ARTIFACT HASHES
    # =================================================================

    source_manifests.append({

        "dataset":
            dataset["name"],

        "path":
            dataset[
                "manifest"
            ]
            .relative_to(
                ROOT
            )
            .as_posix(),

        "sha256":
            sha256(
                dataset[
                    "manifest"
                ]
            ),
    })


    source_metadata.append({

        "dataset":
            dataset["name"],

        "path":
            dataset[
                "metadata"
            ]
            .relative_to(
                ROOT
            )
            .as_posix(),

        "sha256":
            sha256(
                dataset[
                    "metadata"
                ]
            ),
    })


# =====================================================================
# GLOBAL COUNTS
# =====================================================================

real_count = sum(

    row["label"] == 0

    for row in global_rows
)


fake_count = sum(

    row["label"] == 1

    for row in global_rows
)


total_count = len(
    global_rows
)


if real_count != EXPECTED_REAL:

    fail(
        f"Real={real_count}; "
        f"esperaba {EXPECTED_REAL}"
    )


if fake_count != EXPECTED_FAKE:

    fail(
        f"Fake={fake_count}; "
        f"esperaba {EXPECTED_FAKE}"
    )


if total_count != EXPECTED_TOTAL:

    fail(
        f"Total={total_count}; "
        f"esperaba {EXPECTED_TOTAL}"
    )


# =====================================================================
# GENERATOR COUNTS
# =====================================================================

generator_counts = Counter(

    row["generator"]

    for row in global_rows
)


if (
    generator_counts
    != Counter(
        EXPECTED_GENERATOR_COUNTS
    )
):

    fail(
        "Distribución incorrecta:\n\n"
        f"{dict(generator_counts)}"
    )


# =====================================================================
# INTEGRITY COUNTS
# =====================================================================

integrity_counts = Counter(

    row["integrity_status"]

    for row in global_rows
)


expected_verified = 120
expected_anchored = 100


if (
    integrity_counts[
        "VERIFIED_AGAINST_PREVIOUS_FREEZE"
    ]
    != expected_verified
):

    fail(
        "Número incorrecto de WAV verificados "
        "contra hashes anteriores:\n"
        f"{dict(integrity_counts)}"
    )


if (
    integrity_counts[
        "ANCHORED_AT_GLOBAL_FREEZE"
    ]
    != expected_anchored
):

    fail(
        "Número incorrecto de nuevos anclajes "
        "de integridad:\n"
        f"{dict(integrity_counts)}"
    )


# =====================================================================
# DUPLICATE QC
# =====================================================================

hash_groups = defaultdict(
    list
)


for row in global_rows:

    hash_groups[
        row["sha256"]
    ].append(
        row
    )


duplicate_groups = {

    digest:
        rows

    for digest, rows
    in hash_groups.items()

    if len(rows) > 1
}


cross_class_duplicates = []


for digest, rows in duplicate_groups.items():

    classes = {

        row["class"]

        for row in rows
    }


    if len(classes) > 1:

        cross_class_duplicates.append({

            "sha256":
                digest,

            "files":

                [
                    row[
                        "relative_path"
                    ]

                    for row in rows
                ],
        })


if cross_class_duplicates:

    fail(
        "Existen WAV idénticos entre REAL y FAKE:\n\n"
        + json.dumps(
            cross_class_duplicates,
            indent=2,
            ensure_ascii=False
        )
    )


if duplicate_groups:

    warnings.append(
        f"{len(duplicate_groups)} grupos "
        "de WAV duplicados dentro de la misma clase."
    )


# =====================================================================
# SAMPLE RATE SUMMARY
# =====================================================================

sample_rate_counts = Counter(

    row["sample_rate"]

    for row in global_rows
)


# =====================================================================
# FINGERPRINT
#
# This binds:
#
# - path
# - label
# - class
# - generator
# - SHA256
# =====================================================================

fingerprint_lines = []


for row in sorted(

    global_rows,

    key=lambda x:
        x[
            "relative_path"
        ]
):

    fingerprint_lines.append(

        f"{row['relative_path']}|"
        f"{row['label']}|"
        f"{row['class']}|"
        f"{row['generator']}|"
        f"{row['sha256']}"
    )


fingerprint_payload = "\n".join(
    fingerprint_lines
)


dataset_fingerprint = hashlib.sha256(

    fingerprint_payload.encode(
        "utf-8"
    )

).hexdigest()


# =====================================================================
# ALL 220 WAVS HAVE PASSED QC.
# ONLY NOW WRITE GLOBAL FREEZE.
# =====================================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# GLOBAL CSV
# =====================================================================

with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(

        f,

        fieldnames=
            global_rows[
                0
            ].keys()
    )

    writer.writeheader()

    writer.writerows(
        global_rows
    )


# =====================================================================
# GLOBAL MANIFEST
# =====================================================================

global_manifest = {

    "name":
        "External Final Holdout",

    "status":
        "FROZEN",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "root_at_freeze":
        str(ROOT),


    # -----------------------------------------------------------------
    # IMPORTANT METHODOLOGICAL NOTE
    # -----------------------------------------------------------------

    "integrity_policy": {

        "fake_audio":
            (
                "All 120 fake WAVs were verified against "
                "SHA-256 values stored during their previous "
                "generator-specific freeze."
            ),

        "real_voxpopuli":
            (
                "The original VoxPopuli freeze did not store "
                "individual WAV SHA-256 values. Because no detector "
                "evaluation had begun, the current 100 real WAVs "
                "were SHA-256 anchored for the first time during "
                "this global final freeze."
            ),

        "post_freeze_rule":
            (
                "No WAV may be modified, regenerated, deleted, "
                "replaced or selected based on detector results "
                "after this global freeze."
            ),
    },


    # -----------------------------------------------------------------
    # COUNTS
    # -----------------------------------------------------------------

    "counts": {

        "total":
            total_count,

        "real":
            real_count,

        "fake":
            fake_count,

        "by_generator":
            dict(
                generator_counts
            ),

        "by_integrity_status":
            dict(
                integrity_counts
            ),

        "by_sample_rate":

            {

                str(sr):
                    count

                for sr, count
                in sorted(
                    sample_rate_counts.items()
                )
            },
    },


    # -----------------------------------------------------------------
    # FINGERPRINT
    # -----------------------------------------------------------------

    "dataset_fingerprint_sha256":
        dataset_fingerprint,

    "fingerprint_definition":
        (
            "SHA256 of sorted lines: "
            "<relative_path>|<label>|<class>|"
            "<generator>|<wav_sha256>"
        ),


    # -----------------------------------------------------------------
    # GLOBAL CSV
    # -----------------------------------------------------------------

    "metadata_csv": {

        "path":
            OUT_CSV
            .relative_to(
                ROOT
            )
            .as_posix(),

        "sha256":
            sha256(
                OUT_CSV
            ),
    },


    # -----------------------------------------------------------------
    # PREVIOUS ARTIFACTS
    # -----------------------------------------------------------------

    "source_manifests":
        source_manifests,

    "source_metadata":
        source_metadata,


    # -----------------------------------------------------------------
    # DUPLICATE QC
    # -----------------------------------------------------------------

    "duplicate_qc": {

        "duplicate_hash_groups":
            len(
                duplicate_groups
            ),

        "cross_class_duplicates":
            0,
    },


    # -----------------------------------------------------------------
    # FILES
    # -----------------------------------------------------------------

    "files": [

        {

            "relative_path":
                row[
                    "relative_path"
                ],

            "file":
                row[
                    "file"
                ],

            "label":
                row[
                    "label"
                ],

            "class":
                row[
                    "class"
                ],

            "generator":
                row[
                    "generator"
                ],

            "speaker":
                row[
                    "speaker"
                ],

            "sample_rate":
                row[
                    "sample_rate"
                ],

            "channels":
                row[
                    "channels"
                ],

            "duration_s":
                row[
                    "duration_s"
                ],

            "bytes":
                row[
                    "bytes"
                ],

            "sha256":
                row[
                    "sha256"
                ],

            "integrity_status":
                row[
                    "integrity_status"
                ],
        }

        for row in global_rows
    ],
}


with OUT_MANIFEST.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        global_manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 72)
print(
    "EXTERNAL FINAL HOLDOUT — GLOBALMENTE CONGELADO"
)
print("=" * 72)

print(
    f"Real      : {real_count}"
)

print(
    f"Fake      : {fake_count}"
)

print(
    f"Total     : {total_count}"
)


print()
print(
    "Integridad:"
)

print(
    "  Verificados contra freeze previo : "
    f"{integrity_counts['VERIFIED_AGAINST_PREVIOUS_FREEZE']}"
)

print(
    "  Anclados en freeze global        : "
    f"{integrity_counts['ANCHORED_AT_GLOBAL_FREEZE']}"
)


print()
print(
    "Distribución:"
)


for generator in [

    "VoxPopuli",
    "RVC",
    "Qwen3-TTS",
    "Fun-CosyVoice3",
    "Confucius4-TTS",
    "OpenVoice V2",
    "VoxCPM2",

]:

    print(
        f"  {generator:<20}: "
        f"{generator_counts[generator]}"
    )


print()
print(
    "Sample rates:"
)


for sr, count in sorted(
    sample_rate_counts.items()
):

    print(
        f"  {sr:>6} Hz : {count}"
    )


print()
print(
    f"Metadata  : {OUT_CSV}"
)

print(
    f"Manifest  : {OUT_MANIFEST}"
)


print()
print(
    "Fingerprint SHA256:"
)

print(
    dataset_fingerprint
)


if warnings:

    print()
    print("AVISOS:")

    for warning in warnings:

        print(
            " -",
            warning
        )


print()
print(
    "QC         : OK"
)

print()
print(
    "HOLDOUT EXTERNO CERRADO."
)

print(
    "NO modificar, regenerar, eliminar "
    "ni sustituir ningún WAV."
)

print("=" * 72)