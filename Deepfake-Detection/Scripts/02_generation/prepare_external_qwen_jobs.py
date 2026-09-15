from pathlib import Path
import csv
import hashlib
import sys


ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

SELECTION_CSV = (
    ROOT / "_external_fake_build" / "rvc_selection.csv"
)

REF_DIR = (
    ROOT / "_external_fake_build" / "rvc_sources"
)

VOX_METADATA = (
    ROOT
    / "external_final_voxpopuli"
    / "metadata_voxpopuli_real.csv"
)

OUT_CSV = (
    ROOT
    / "_external_fake_build"
    / "qwen_jobs.csv"
)

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGE = "Spanish"

# 20 textos completamente fijados.
# Estos mismos textos deberán reutilizarse para los demás TTS.
TARGET_TEXTS = [
    "La reunión comenzará a las nueve de la mañana en la sala principal.",
    "El tren llegó a la estación unos minutos antes de lo previsto.",
    "Durante la tarde esperamos cielos despejados y temperaturas suaves.",
    "La biblioteca permanecerá abierta hasta las ocho de la noche.",
    "Mañana revisaremos los resultados obtenidos durante la última prueba.",
    "El sistema procesa cada archivo de manera independiente y ordenada.",
    "La carretera atraviesa varios pueblos antes de llegar a la ciudad.",
    "Necesitamos comprobar que todos los datos se hayan guardado correctamente.",
    "El equipo terminó el proyecto después de varias semanas de trabajo.",
    "La aplicación permite consultar la información desde cualquier dispositivo.",
    "Esta mañana encontré las llaves encima de la mesa del salón.",
    "Los participantes recibieron las instrucciones antes de comenzar la actividad.",
    "El informe incluye una explicación detallada de todos los resultados.",
    "La próxima semana tendremos una nueva sesión para revisar el progreso.",
    "El archivo contiene únicamente la información necesaria para realizar la prueba.",
    "La universidad publicó ayer el calendario definitivo del próximo curso.",
    "Después de analizar los datos podremos comparar los diferentes métodos.",
    "La grabación se realizó en una habitación tranquila y sin ruido de fondo.",
    "Cada muestra debe conservarse sin modificaciones después de la evaluación.",
    "Los resultados finales se utilizarán para estudiar la capacidad de generalización.",
]


def find_column(fieldnames, candidates):
    normalized = {
        x.lower().strip(): x
        for x in fieldnames
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


if OUT_CSV.exists():
    sys.exit(
        f"ERROR: ya existe {OUT_CSV}\n"
        "No se sobrescribe automáticamente."
    )


# ============================================================
# SELECCIÓN EXTERNA
# ============================================================

with SELECTION_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    selection = list(csv.DictReader(f))


if len(selection) != 20:
    sys.exit(
        f"ERROR: esperaba 20 referencias, "
        f"pero hay {len(selection)}."
    )


# ============================================================
# METADATA VOXPOPULI
# ============================================================

with VOX_METADATA.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    fields = reader.fieldnames or []

    file_col = find_column(
        fields,
        [
            "file",
            "filename",
            "wav",
            "wav_file",
            "audio",
            "audio_path",
            "path",
        ]
    )

    text_col = find_column(
        fields,
        [
            "text",
            "sentence",
            "transcript",
            "transcription",
            "normalized_text",
            "raw_text",
        ]
    )

    if file_col is None:
        print("Columnas:", fields)
        sys.exit(
            "ERROR: no encuentro columna de fichero."
        )

    if text_col is None:
        print("Columnas:", fields)
        sys.exit(
            "ERROR: no encuentro transcripción."
        )

    metadata = list(reader)


# Índice por basename
text_by_file = {}

for row in metadata:

    name = Path(
        str(row[file_col]).strip()
    ).name

    text = str(
        row[text_col]
    ).strip()

    if name and text:
        text_by_file[name] = text


# ============================================================
# CREAR JOBS
# ============================================================

jobs = []

selection = sorted(
    selection,
    key=lambda x: x["rvc_input"]
)


for index, row in enumerate(selection):

    ref_file = REF_DIR / row["rvc_input"]

    original_file = row["original_file"]

    if not ref_file.exists():
        sys.exit(
            f"ERROR: falta {ref_file}"
        )

    ref_text = text_by_file.get(
        original_file
    )

    if not ref_text:
        sys.exit(
            "ERROR: no encuentro transcript para "
            f"{original_file}"
        )

    target_text = TARGET_TEXTS[index]

    jobs.append({
        "id": f"qwen_{index + 1:03d}",
        "speaker": row["speaker"],

        "reference_file": str(ref_file),
        "original_reference_file": original_file,

        "reference_text": ref_text,

        "target_text": target_text,

        "language": LANGUAGE,
        "model": MODEL,

        "clone_mode": "ICL",
        "x_vector_only_mode": False,

        "target_text_sha256": sha256_text(
            target_text
        ),

        # Seed individual y fijo
        "seed": 2026091501 + index,
    })


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
print("QWEN EXTERNAL JOBS — PREPARADOS")
print("=" * 72)

print(f"Model      : {MODEL}")
print(f"Language   : {LANGUAGE}")
print(f"Clone mode : ICL")
print(f"Speakers   : {len(set(x['speaker'] for x in jobs))}")
print(f"Jobs       : {len(jobs)}")
print()
print(f"CSV        : {OUT_CSV}")
print()
print("NO se ha generado audio todavía.")
print("=" * 72)