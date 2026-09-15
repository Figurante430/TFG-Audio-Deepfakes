from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SR = 16000
FRAME_MS = 25.0
HOP_MS = 10.0
SILENCE_DBFS = -45.0

PROBLEM_SPEAKERS = {"speaker_ext_02", "speaker_ext_03"}
CONTROL_SPEAKERS = {"speaker_ext_01", "speaker_ext_04", "speaker_ext_05"}


def dbfs(x: float, floor: float = -120.0) -> float:
    if x <= 0:
        return floor
    return max(20.0 * math.log10(x), floor)


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def frame_signal(x: np.ndarray, frame_len: int, hop_len: int):
    if len(x) < frame_len:
        padded = np.pad(x, (0, frame_len - len(x)))
        return padded[None, :]

    n_frames = 1 + (len(x) - frame_len) // hop_len
    shape = (n_frames, frame_len)
    strides = (x.strides[0] * hop_len, x.strides[0])

    return np.lib.stride_tricks.as_strided(
        x,
        shape=shape,
        strides=strides,
        writeable=False,
    )


def spectral_features(x: np.ndarray, sr: int):
    # Para no depender de scipy/librosa, usamos un FFT global con ventana Hann.
    if len(x) < 2:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "rolloff95_hz": 0.0,
            "hf_energy_ratio_4k": 0.0,
            "lf_energy_ratio_300": 0.0,
            "spectral_flatness": 0.0,
        }

    window = np.hanning(len(x)).astype(np.float32)
    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)

    total = float(power.sum())

    if total <= 1e-20:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "rolloff95_hz": 0.0,
            "hf_energy_ratio_4k": 0.0,
            "lf_energy_ratio_300": 0.0,
            "spectral_flatness": 0.0,
        }

    centroid = float((freqs * power).sum() / total)
    bandwidth = float(
        np.sqrt((((freqs - centroid) ** 2) * power).sum() / total)
    )

    cumulative = np.cumsum(power)
    roll_idx = int(np.searchsorted(cumulative, 0.95 * total))
    roll_idx = min(roll_idx, len(freqs) - 1)
    rolloff95 = float(freqs[roll_idx])

    hf_ratio = float(power[freqs >= 4000].sum() / total)
    lf_ratio = float(power[freqs <= 300].sum() / total)

    eps = 1e-20
    positive = power + eps
    flatness = float(
        np.exp(np.mean(np.log(positive))) / np.mean(positive)
    )

    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "rolloff95_hz": rolloff95,
        "hf_energy_ratio_4k": hf_ratio,
        "lf_energy_ratio_300": lf_ratio,
        "spectral_flatness": flatness,
    }


def leading_trailing_silence_ms(
    frame_db: np.ndarray,
    hop_ms: float,
    threshold_db: float,
):
    active = frame_db > threshold_db

    if not np.any(active):
        total = len(frame_db) * hop_ms
        return total, total

    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))

    leading = first * hop_ms
    trailing = (len(active) - 1 - last) * hop_ms

    return float(leading), float(trailing)


def audio_features(path: Path):
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = x.mean(axis=1)

    x = np.asarray(x, dtype=np.float32)

    if sr != TARGET_SR:
        raise RuntimeError(
            f"{path} está a {sr} Hz; esperaba {TARGET_SR} Hz."
        )

    duration = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    mean_abs = float(np.mean(np.abs(x))) if len(x) else 0.0
    dc = float(np.mean(x)) if len(x) else 0.0

    clipping_ratio = (
        float(np.mean(np.abs(x) >= 0.999))
        if len(x)
        else 0.0
    )

    near_zero_ratio = (
        float(np.mean(np.abs(x) <= 1e-4))
        if len(x)
        else 1.0
    )

    if rms > 0:
        crest_db = dbfs(peak / rms)
    else:
        crest_db = 0.0

    if len(x) > 1:
        zcr = float(
            np.mean(
                np.signbit(x[1:]) != np.signbit(x[:-1])
            )
        )
    else:
        zcr = 0.0

    frame_len = max(1, int(sr * FRAME_MS / 1000.0))
    hop_len = max(1, int(sr * HOP_MS / 1000.0))

    frames = frame_signal(x, frame_len, hop_len)

    frame_rms = np.sqrt(
        np.mean(frames * frames, axis=1) + 1e-20
    )

    frame_db = 20.0 * np.log10(frame_rms + 1e-20)

    silence_ratio = float(
        np.mean(frame_db <= SILENCE_DBFS)
    )

    p10 = float(np.percentile(frame_db, 10))
    p20 = float(np.percentile(frame_db, 20))
    p50 = float(np.percentile(frame_db, 50))
    p80 = float(np.percentile(frame_db, 80))
    p90 = float(np.percentile(frame_db, 90))
    p95 = float(np.percentile(frame_db, 95))

    # No es una estimación física de SNR, sino un proxy robusto:
    # diferencia entre frames de alta y baja energía.
    snr_proxy = p80 - p20
    dynamic_range = p95 - p10

    leading_ms, trailing_ms = leading_trailing_silence_ms(
        frame_db,
        HOP_MS,
        SILENCE_DBFS,
    )

    spec = spectral_features(x, sr)

    return {
        "duration_s": duration,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "mean_abs": mean_abs,
        "dc_offset": dc,
        "crest_factor_db": crest_db,
        "clipping_ratio": clipping_ratio,
        "near_zero_ratio": near_zero_ratio,
        "zero_crossing_rate": zcr,
        "silence_ratio": silence_ratio,
        "leading_silence_ms": leading_ms,
        "trailing_silence_ms": trailing_ms,
        "frame_db_p10": p10,
        "frame_db_p50": p50,
        "frame_db_p90": p90,
        "dynamic_range_proxy_db": dynamic_range,
        "snr_proxy_db": snr_proxy,
        **spec,
    }


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(arr))

    if len(arr) > 1:
        std = float(np.std(arr, ddof=1))
    else:
        std = 0.0

    return mean, std


def rankdata(values):
    """
    Ranks medios para empates, sin scipy.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while (
            j < len(values)
            and values[order[j]] == values[order[i]]
        ):
            j += 1

        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan"), float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0

    pearson = float(np.corrcoef(x, y)[0, 1])

    xr = rankdata(x)
    yr = rankdata(y)

    spearman = float(np.corrcoef(xr, yr)[0, 1])

    return pearson, spearman


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)

    pooled_num = (
        (len(a) - 1) * s1
        + (len(b) - 1) * s2
    )
    pooled_den = len(a) + len(b) - 2

    if pooled_den <= 0:
        return float("nan")

    pooled = math.sqrt(pooled_num / pooled_den)

    if pooled == 0:
        return 0.0

    return float(
        (np.mean(a) - np.mean(b)) / pooled
    )


FEATURE_NAMES = [
    "duration_s",
    "rms_dbfs",
    "peak_dbfs",
    "mean_abs",
    "dc_offset",
    "crest_factor_db",
    "clipping_ratio",
    "near_zero_ratio",
    "zero_crossing_rate",
    "silence_ratio",
    "leading_silence_ms",
    "trailing_silence_ms",
    "frame_db_p10",
    "frame_db_p50",
    "frame_db_p90",
    "dynamic_range_proxy_db",
    "snr_proxy_db",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "rolloff95_hz",
    "hf_energy_ratio_4k",
    "lf_energy_ratio_300",
    "spectral_flatness",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"C:\Users\gonza\TFG\Deepfake-Detection"
        ),
    )

    args = parser.parse_args()

    root = args.root

    real_dir = (
        root
        / "external_test"
        / "real"
    )

    consensus_path = (
        root
        / "external_test"
        / "logo_wavlm_21_external_real"
        / "consensus_by_file.csv"
    )

    out_dir = (
        root
        / "external_test"
        / "acoustic_analysis"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not consensus_path.exists():
        raise FileNotFoundError(
            f"No existe:\n{consensus_path}"
        )

    consensus_rows = read_csv(
        consensus_path
    )

    consensus_map = {}

    for row in consensus_rows:
        key = (
            row["speaker_id"],
            row["filename"],
        )

        consensus_map[key] = row

    wavs = sorted(
        real_dir.rglob("*.wav")
    )

    if not wavs:
        raise FileNotFoundError(
            f"No hay WAV en {real_dir}"
        )

    print(
        f"Analizando {len(wavs)} audios externos..."
    )

    feature_rows = []

    for i, path in enumerate(wavs, 1):
        speaker = path.parent.name
        key = (speaker, path.name)

        consensus = consensus_map.get(key)

        if consensus is None:
            raise RuntimeError(
                "No encuentro este audio en consensus_by_file.csv:\n"
                f"{path}"
            )

        feats = audio_features(path)

        row = {
            "path": str(path),
            "filename": path.name,
            "speaker_id": speaker,
            "n_models": int(consensus["n_models"]),
            "n_pred_fake": int(consensus["n_pred_fake"]),
            "fake_vote_rate": safe_float(
                consensus["fake_vote_rate"]
            ),
            "prob_fake_mean": safe_float(
                consensus["prob_fake_mean"]
            ),
            "prob_fake_median": safe_float(
                consensus["prob_fake_median"]
            ),
            "prob_fake_min": safe_float(
                consensus["prob_fake_min"]
            ),
            "prob_fake_max": safe_float(
                consensus["prob_fake_max"]
            ),
            **feats,
        }

        feature_rows.append(row)

        print(
            f"\rProcesados: {i}/{len(wavs)}",
            end="",
            flush=True,
        )

    print()

    write_csv(
        out_dir / "features_external_real.csv",
        feature_rows,
    )

    # --------------------------------------------------------
    # Resumen por speaker
    # --------------------------------------------------------
    by_speaker = defaultdict(list)

    for row in feature_rows:
        by_speaker[row["speaker_id"]].append(row)

    speaker_summary = []

    for speaker in sorted(by_speaker):
        rows = by_speaker[speaker]

        base = {
            "speaker_id": speaker,
            "n": len(rows),
            "fake_vote_rate_mean": float(
                np.mean(
                    [
                        r["fake_vote_rate"]
                        for r in rows
                    ]
                )
            ),
            "prob_fake_mean": float(
                np.mean(
                    [
                        r["prob_fake_mean"]
                        for r in rows
                    ]
                )
            ),
        }

        for feature in FEATURE_NAMES:
            mean, std = mean_std(
                [
                    r[feature]
                    for r in rows
                ]
            )

            base[f"{feature}_mean"] = mean
            base[f"{feature}_std"] = std

        speaker_summary.append(base)

    write_csv(
        out_dir / "summary_by_speaker.csv",
        speaker_summary,
    )

    # --------------------------------------------------------
    # Correlación feature vs fake_vote_rate
    # --------------------------------------------------------
    corr_rows = []

    target = [
        r["fake_vote_rate"]
        for r in feature_rows
    ]

    for feature in FEATURE_NAMES:
        values = [
            r[feature]
            for r in feature_rows
        ]

        pearson, spearman = correlation(
            values,
            target,
        )

        corr_rows.append(
            {
                "feature": feature,
                "pearson_vs_fake_vote_rate": pearson,
                "spearman_vs_fake_vote_rate": spearman,
                "abs_spearman": abs(spearman),
            }
        )

    corr_rows.sort(
        key=lambda r: r["abs_spearman"],
        reverse=True,
    )

    write_csv(
        out_dir / "feature_correlations_with_fake_vote.csv",
        corr_rows,
    )

    # --------------------------------------------------------
    # Speakers problemáticos (02+03) vs control (01+04+05)
    # --------------------------------------------------------
    problem_rows = [
        r for r in feature_rows
        if r["speaker_id"] in PROBLEM_SPEAKERS
    ]

    control_rows = [
        r for r in feature_rows
        if r["speaker_id"] in CONTROL_SPEAKERS
    ]

    effect_rows = []

    for feature in FEATURE_NAMES:
        a = [
            r[feature]
            for r in problem_rows
        ]

        b = [
            r[feature]
            for r in control_rows
        ]

        mean_a, std_a = mean_std(a)
        mean_b, std_b = mean_std(b)

        d = cohens_d(a, b)

        effect_rows.append(
            {
                "feature": feature,
                "problem_mean": mean_a,
                "problem_std": std_a,
                "control_mean": mean_b,
                "control_std": std_b,
                "cohens_d_problem_minus_control": d,
                "abs_cohens_d": abs(d)
                if math.isfinite(d)
                else float("nan"),
            }
        )

    effect_rows.sort(
        key=lambda r: (
            -1
            if not math.isfinite(r["abs_cohens_d"])
            else r["abs_cohens_d"]
        ),
        reverse=True,
    )

    write_csv(
        out_dir / "problem_vs_control_effects.csv",
        effect_rows,
    )

    # --------------------------------------------------------
    # Top archivos que más modelos marcan como fake
    # --------------------------------------------------------
    suspicious = sorted(
        feature_rows,
        key=lambda r: (
            r["fake_vote_rate"],
            r["prob_fake_mean"],
        ),
        reverse=True,
    )

    write_csv(
        out_dir / "top_suspicious_files.csv",
        suspicious[:30],
    )

    # --------------------------------------------------------
    # Reporte legible
    # --------------------------------------------------------
    report_path = out_dir / "report.txt"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "ANÁLISIS ACÚSTICO DEL EXTERNAL REAL TEST\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(
            f"Audios analizados: {len(feature_rows)}\n"
        )
        f.write(
            "Speakers problemáticos: speaker_ext_02 + speaker_ext_03\n"
        )
        f.write(
            "Control: speaker_ext_01 + speaker_ext_04 + speaker_ext_05\n\n"
        )

        f.write(
            "FAKE VOTE RATE MEDIO POR SPEAKER\n"
        )
        f.write("-" * 72 + "\n")

        for speaker in sorted(by_speaker):
            rows = by_speaker[speaker]

            vote_mean = float(
                np.mean(
                    [
                        r["fake_vote_rate"]
                        for r in rows
                    ]
                )
            )

            prob_mean = float(
                np.mean(
                    [
                        r["prob_fake_mean"]
                        for r in rows
                    ]
                )
            )

            f.write(
                f"{speaker:18s} "
                f"vote_fake={vote_mean:.4f} "
                f"prob_fake_mean={prob_mean:.4f}\n"
            )

        f.write("\n")
        f.write(
            "TOP 10 CORRELACIONES CON FAKE VOTE RATE\n"
        )
        f.write("-" * 72 + "\n")

        for row in corr_rows[:10]:
            f.write(
                f"{row['feature']:28s} "
                f"Pearson={row['pearson_vs_fake_vote_rate']:+.4f} "
                f"Spearman={row['spearman_vs_fake_vote_rate']:+.4f}\n"
            )

        f.write("\n")
        f.write(
            "TOP 10 DIFERENCIAS 02+03 VS CONTROL (COHEN'S d)\n"
        )
        f.write("-" * 72 + "\n")

        for row in effect_rows[:10]:
            f.write(
                f"{row['feature']:28s} "
                f"problem={row['problem_mean']:.4f} "
                f"control={row['control_mean']:.4f} "
                f"d={row['cohens_d_problem_minus_control']:+.4f}\n"
            )

        f.write("\n")
        f.write(
            "TOP 15 ARCHIVOS MÁS SOSPECHOSOS PARA LOS 21 MODELOS\n"
        )
        f.write("-" * 72 + "\n")

        for row in suspicious[:15]:
            f.write(
                f"{row['speaker_id']:18s} "
                f"{row['filename']:28s} "
                f"fake_votes={row['n_pred_fake']:2d}/{row['n_models']:2d} "
                f"vote_rate={row['fake_vote_rate']:.4f} "
                f"prob_mean={row['prob_fake_mean']:.4f}\n"
            )

        f.write("\n")
        f.write(
            "NOTA: snr_proxy_db y dynamic_range_proxy_db son proxies basados\n"
        )
        f.write(
            "en energía por frames, no estimaciones físicas de SNR/dinámica.\n"
        )

    print()
    print("=" * 72)
    print("ANÁLISIS TERMINADO")
    print("=" * 72)
    print()
    print("Top correlaciones con fake_vote_rate:")

    for row in corr_rows[:8]:
        print(
            f"  {row['feature']:28s} "
            f"rho={row['spearman_vs_fake_vote_rate']:+.4f}"
        )

    print()
    print("Mayores diferencias speaker 02+03 vs control:")

    for row in effect_rows[:8]:
        print(
            f"  {row['feature']:28s} "
            f"d={row['cohens_d_problem_minus_control']:+.4f}"
        )

    print()
    print("Archivos creados en:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
