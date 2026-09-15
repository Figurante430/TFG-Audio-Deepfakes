from __future__ import annotations

import csv
import math
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")

DATA_DIR = ROOT / "data_normalized"

METADATA_WITH_SPLIT = DATA_DIR / "metadata_normalized.csv"
METADATA_CSV = DATA_DIR / "metadata_normalized.csv"

REPORT_CSV = DATA_DIR / "validation_report.csv"
BAD_CSV = DATA_DIR / "validation_bad_files.csv"
SUMMARY_TXT = DATA_DIR / "validation_summary.txt"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2   # PCM16
MIN_DURATION_SECONDS = 0.50
MAX_DURATION_SECONDS = 30.0

# Si más del 0.5% de las muestras están saturadas, lo marcamos como aviso
MAX_CLIPPING_PERCENT = 0.50


# ============================================================
# UTILIDADES
# ============================================================

def load_metadata():
    if METADATA_WITH_SPLIT.exists():
        metadata_path = METADATA_WITH_SPLIT
    elif METADATA_CSV.exists():
        metadata_path = METADATA_CSV
    else:
        raise FileNotFoundError(
            "No encuentro ni metadata_with_split.csv ni metadata.csv en:\n"
            f"{DATA_DIR}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"{metadata_path.name} está vacío.")

    if "path" not in rows[0]:
        raise RuntimeError(
            f"{metadata_path.name} no contiene la columna 'path'."
        )

    print(f"Metadata usada: {metadata_path}")
    print(f"Filas: {len(rows)}")

    return metadata_path, rows


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)

    if p.is_absolute():
        return p

    return ROOT / p


def pcm16_metrics(raw_frames: bytes):
    """
    Calcula peak, RMS y clipping sin NumPy.
    Solo para PCM16 little-endian.
    """
    if not raw_frames:
        return None, None, None

    # Cada sample PCM16 ocupa 2 bytes
    n_samples = len(raw_frames) // 2

    if n_samples == 0:
        return None, None, None

    sum_sq = 0
    peak = 0
    clipped = 0

    for i in range(0, n_samples * 2, 2):
        value = int.from_bytes(
            raw_frames[i:i+2],
            byteorder="little",
            signed=True,
        )

        abs_value = abs(value)

        if abs_value > peak:
            peak = abs_value

        sum_sq += value * value

        # Cerca del máximo de PCM16
        if abs_value >= 32760:
            clipped += 1

    rms = math.sqrt(sum_sq / n_samples)

    peak_norm = peak / 32768.0
    rms_norm = rms / 32768.0

    if rms_norm > 0:
        rms_dbfs = 20.0 * math.log10(rms_norm)
    else:
        rms_dbfs = float("-inf")

    clipping_percent = (clipped / n_samples) * 100.0

    return peak_norm, rms_dbfs, clipping_percent


# ============================================================
# VALIDACIÓN DE UN WAV
# ============================================================

def inspect_wav(path: Path):
    result = {
        "exists": False,
        "readable": False,
        "sample_rate": "",
        "channels": "",
        "sample_width_bytes": "",
        "frames": "",
        "duration_seconds": "",
        "compression": "",
        "peak": "",
        "rms_dbfs": "",
        "clipping_percent": "",
        "valid_sr": False,
        "valid_channels": False,
        "valid_pcm16": False,
        "valid_duration": False,
        "status": "",
        "issues": "",
    }

    issues = []

    if not path.exists():
        result["status"] = "ERROR"
        result["issues"] = "FILE_NOT_FOUND"
        return result

    result["exists"] = True

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            compression = wf.getcomptype()

            duration = (
                frames / sample_rate
                if sample_rate > 0
                else 0.0
            )

            raw_frames = wf.readframes(frames)

        result["readable"] = True
        result["sample_rate"] = sample_rate
        result["channels"] = channels
        result["sample_width_bytes"] = sample_width
        result["frames"] = frames
        result["duration_seconds"] = round(duration, 4)
        result["compression"] = compression

        result["valid_sr"] = sample_rate == EXPECTED_SAMPLE_RATE
        result["valid_channels"] = channels == EXPECTED_CHANNELS
        result["valid_pcm16"] = (
            sample_width == EXPECTED_SAMPLE_WIDTH_BYTES
            and compression == "NONE"
        )
        result["valid_duration"] = (
            MIN_DURATION_SECONDS
            <= duration
            <= MAX_DURATION_SECONDS
        )

        if not result["valid_sr"]:
            issues.append(f"SR_{sample_rate}")

        if not result["valid_channels"]:
            issues.append(f"CHANNELS_{channels}")

        if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
            issues.append(f"SAMPLE_WIDTH_{sample_width}")

        if compression != "NONE":
            issues.append(f"COMPRESSION_{compression}")

        if duration < MIN_DURATION_SECONDS:
            issues.append("TOO_SHORT")

        if duration > MAX_DURATION_SECONDS:
            issues.append("TOO_LONG")

        # Métricas de amplitud solo si es PCM16
        if sample_width == 2 and compression == "NONE":
            peak, rms_dbfs, clipping_percent = pcm16_metrics(raw_frames)

            if peak is not None:
                result["peak"] = round(peak, 6)

            if rms_dbfs is not None:
                if math.isinf(rms_dbfs):
                    result["rms_dbfs"] = "-inf"
                else:
                    result["rms_dbfs"] = round(rms_dbfs, 3)

            if clipping_percent is not None:
                result["clipping_percent"] = round(
                    clipping_percent,
                    4
                )

                if clipping_percent > MAX_CLIPPING_PERCENT:
                    issues.append("CLIPPING")

            if peak == 0:
                issues.append("SILENT_AUDIO")

    except Exception as e:
        result["status"] = "ERROR"
        result["issues"] = (
            f"UNREADABLE: {type(e).__name__}: {e}"
        )
        return result

    if issues:
        result["status"] = "WARN"
        result["issues"] = ";".join(issues)
    else:
        result["status"] = "OK"
        result["issues"] = ""

    return result


# ============================================================
# VALIDACIÓN GLOBAL
# ============================================================

def validate_dataset(metadata_rows):
    report_rows = []

    total = len(metadata_rows)

    for i, row in enumerate(metadata_rows, 1):
        path = resolve_path(row["path"])

        print(
            f"[{i:04d}/{total}] {path.name}",
            end="\r"
        )

        info = inspect_wav(path)

        report = dict(row)
        report["resolved_path"] = str(path)

        for key, value in info.items():
            report[key] = value

        report_rows.append(report)

    print()
    return report_rows


# ============================================================
# VALIDACIONES DE METADATA / SPLITS
# ============================================================

def validate_metadata_consistency(rows):
    messages = []

    # Rutas duplicadas
    path_counts = Counter(row["path"] for row in rows)

    duplicate_paths = [
        p for p, c in path_counts.items()
        if c > 1
    ]

    if duplicate_paths:
        messages.append(
            f"[WARN] Rutas duplicadas: {len(duplicate_paths)}"
        )
    else:
        messages.append(
            "[OK] No hay rutas duplicadas."
        )

    # Leakage de utterance_id si existe split
    if (
        rows
        and "split" in rows[0]
        and "utterance_id" in rows[0]
    ):
        split_by_utt = defaultdict(set)

        for row in rows:
            split_by_utt[row["utterance_id"]].add(
                row["split"]
            )

        leaked = {
            utt: splits
            for utt, splits in split_by_utt.items()
            if len(splits) > 1
        }

        if leaked:
            messages.append(
                f"[ERROR] Leakage de utterance_id: "
                f"{len(leaked)} IDs aparecen en varios splits."
            )
        else:
            messages.append(
                "[OK] No hay leakage de utterance_id entre splits."
            )

    return messages


# ============================================================
# RESUMEN
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def build_summary(report_rows, metadata_messages):
    lines = []

    lines.append("=" * 72)
    lines.append("VALIDACIÓN DEL DATASET")
    lines.append("=" * 72)

    total = len(report_rows)

    ok = sum(r["status"] == "OK" for r in report_rows)
    warn = sum(r["status"] == "WARN" for r in report_rows)
    error = sum(r["status"] == "ERROR" for r in report_rows)

    lines.append("")
    lines.append(f"TOTAL : {total}")
    lines.append(f"OK    : {ok}")
    lines.append(f"WARN  : {warn}")
    lines.append(f"ERROR : {error}")

    durations = [
        safe_float(r["duration_seconds"])
        for r in report_rows
    ]
    durations = [
        x for x in durations
        if x is not None
    ]

    if durations:
        lines.append("")
        lines.append("DURACIÓN")
        lines.append(
            f"  mínima : {min(durations):.3f} s"
        )
        lines.append(
            f"  máxima : {max(durations):.3f} s"
        )
        lines.append(
            f"  media  : {statistics.mean(durations):.3f} s"
        )
        lines.append(
            f"  mediana: {statistics.median(durations):.3f} s"
        )
        lines.append(
            f"  total  : {sum(durations)/60:.2f} min"
        )

    lines.append("")
    lines.append("SAMPLE RATE")
    for sr, count in sorted(
        Counter(
            str(r["sample_rate"])
            for r in report_rows
            if r["sample_rate"] != ""
        ).items()
    ):
        lines.append(
            f"  {sr:10s} {count}"
        )

    lines.append("")
    lines.append("CANALES")
    for channels, count in sorted(
        Counter(
            str(r["channels"])
            for r in report_rows
            if r["channels"] != ""
        ).items()
    ):
        lines.append(
            f"  {channels:10s} {count}"
        )

    lines.append("")
    lines.append("SAMPLE WIDTH")
    for width, count in sorted(
        Counter(
            str(r["sample_width_bytes"])
            for r in report_rows
            if r["sample_width_bytes"] != ""
        ).items()
    ):
        lines.append(
            f"  {width + ' bytes':10s} {count}"
        )

    # Distribuciones
    for field, title in [
        ("label", "POR LABEL"),
        ("category", "POR CATEGORÍA"),
        ("generator", "POR GENERADOR"),
        ("speaker_id", "POR SPEAKER FINAL"),
        ("split", "POR SPLIT"),
    ]:
        if report_rows and field in report_rows[0]:
            lines.append("")
            lines.append(title)

            for name, count in sorted(
                Counter(
                    str(r.get(field, ""))
                    for r in report_rows
                ).items()
            ):
                lines.append(
                    f"  {name:18s} {count}"
                )

    # Problemas
    issue_counter = Counter()

    for row in report_rows:
        issues = str(row.get("issues", "")).strip()

        if not issues:
            continue

        for issue in issues.split(";"):
            issue_counter[issue] += 1

    lines.append("")
    lines.append("PROBLEMAS DETECTADOS")

    if issue_counter:
        for issue, count in issue_counter.most_common():
            lines.append(
                f"  {issue:30s} {count}"
            )
    else:
        lines.append("  Ninguno.")

    lines.append("")
    lines.append("METADATA / SPLITS")
    lines.extend(
        f"  {msg}"
        for msg in metadata_messages
    )

    return "\n".join(lines)


# ============================================================
# GUARDADO
# ============================================================

def save_csv(path, rows):
    if not rows:
        return

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


def save_outputs(report_rows, summary_text):
    save_csv(
        REPORT_CSV,
        report_rows
    )

    bad_rows = [
        row for row in report_rows
        if row["status"] != "OK"
    ]

    if bad_rows:
        save_csv(
            BAD_CSV,
            bad_rows
        )
    elif BAD_CSV.exists():
        BAD_CSV.unlink()

    SUMMARY_TXT.write_text(
        summary_text,
        encoding="utf-8"
    )

    print()
    print("Archivos generados:")
    print(f"  {REPORT_CSV}")
    print(f"  {SUMMARY_TXT}")

    if bad_rows:
        print(f"  {BAD_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    _, metadata_rows = load_metadata()

    metadata_messages = validate_metadata_consistency(
        metadata_rows
    )

    report_rows = validate_dataset(
        metadata_rows
    )

    summary = build_summary(
        report_rows,
        metadata_messages
    )

    print()
    print(summary)

    save_outputs(
        report_rows,
        summary
    )


if __name__ == "__main__":
    main()
