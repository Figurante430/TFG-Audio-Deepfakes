from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
SEEDS = [42, 123, 2026]
MODES = ["control", "gate_only"]

EVAL_SCRIPT = ROOT / "evaluar_external_real_wavlm.py"
OUT_ROOT = ROOT / "external_test" / "compare_control_gate_3seeds"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(map(float, values))
    return sum(values) / len(values)


def sample_std(values):
    values = list(map(float, values))
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (
        sum((x - m) ** 2 for x in values)
        / (len(values) - 1)
    ) ** 0.5


def evaluate_run(mode: str, seed: int):
    exp_dir = (
        ROOT
        / "experiments"
        / "wavlm_domain_robust"
        / mode
        / f"seed_{seed}"
    )

    metrics_json = exp_dir / "metrics.json"
    checkpoint = exp_dir / "best_model.pt"

    if not metrics_json.exists():
        raise FileNotFoundError(f"No existe:\n{metrics_json}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"No existe:\n{checkpoint}")

    metrics = read_json(metrics_json)
    threshold = float(metrics["calibrated_threshold"])

    out_csv = (
        OUT_ROOT
        / mode
        / f"seed_{seed}"
        / "predictions.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"{mode.upper()} | seed={seed} | threshold={threshold:.9f}")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--threshold",
        str(threshold),
        "--output",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True)

    rows = read_csv(out_csv)
    n = len(rows)
    tn = sum(int(r["pred_label"]) == 0 for r in rows)
    fp = n - tn
    spec = tn / n
    fpr = fp / n

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    speaker_rows = []
    for speaker in sorted(by_speaker):
        srows = by_speaker[speaker]
        sn = len(srows)
        stn = sum(int(r["pred_label"]) == 0 for r in srows)
        sfp = sn - stn
        speaker_rows.append(
            {
                "mode": mode,
                "seed": seed,
                "speaker_id": speaker,
                "n": sn,
                "tn": stn,
                "fp": sfp,
                "specificity": stn / sn,
                "fpr": sfp / sn,
            }
        )

    internal = metrics["test_threshold_calibrated"]

    summary = {
        "mode": mode,
        "seed": seed,
        "threshold": threshold,
        "internal_accuracy": internal["accuracy"],
        "internal_f1": internal["f1"],
        "internal_recall": internal["recall"],
        "internal_specificity": internal["specificity"],
        "internal_auc": internal["roc_auc"],
        "external_n": n,
        "external_tn": tn,
        "external_fp": fp,
        "external_specificity": spec,
        "external_fpr": fpr,
    }

    return summary, speaker_rows


def main():
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"No existe:\n{EVAL_SCRIPT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_runs = []
    all_speakers = []

    for mode in MODES:
        for seed in SEEDS:
            summary, speaker_rows = evaluate_run(mode, seed)
            all_runs.append(summary)
            all_speakers.extend(speaker_rows)

    write_csv(OUT_ROOT / "summary_all_runs.csv", all_runs)
    write_csv(
        OUT_ROOT / "summary_by_speaker_all_runs.csv",
        all_speakers,
    )

    mode_summary = []
    for mode in MODES:
        rows = [r for r in all_runs if r["mode"] == mode]
        mode_summary.append(
            {
                "mode": mode,
                "n_seeds": len(rows),
                "internal_accuracy_mean": mean(
                    r["internal_accuracy"] for r in rows
                ),
                "internal_accuracy_std": sample_std(
                    r["internal_accuracy"] for r in rows
                ),
                "internal_f1_mean": mean(
                    r["internal_f1"] for r in rows
                ),
                "internal_f1_std": sample_std(
                    r["internal_f1"] for r in rows
                ),
                "external_specificity_mean": mean(
                    r["external_specificity"] for r in rows
                ),
                "external_specificity_std": sample_std(
                    r["external_specificity"] for r in rows
                ),
                "external_fpr_mean": mean(
                    r["external_fpr"] for r in rows
                ),
                "external_fpr_std": sample_std(
                    r["external_fpr"] for r in rows
                ),
            }
        )

    write_csv(OUT_ROOT / "summary_mean_std.csv", mode_summary)

    speaker_summary = []
    speakers = sorted({r["speaker_id"] for r in all_speakers})

    for mode in MODES:
        for speaker in speakers:
            rows = [
                r for r in all_speakers
                if r["mode"] == mode and r["speaker_id"] == speaker
            ]
            speaker_summary.append(
                {
                    "mode": mode,
                    "speaker_id": speaker,
                    "n_seeds": len(rows),
                    "specificity_mean": mean(
                        r["specificity"] for r in rows
                    ),
                    "specificity_std": sample_std(
                        r["specificity"] for r in rows
                    ),
                    "fpr_mean": mean(
                        r["fpr"] for r in rows
                    ),
                    "fpr_std": sample_std(
                        r["fpr"] for r in rows
                    ),
                }
            )

    write_csv(
        OUT_ROOT / "summary_speakers_mean_std.csv",
        speaker_summary,
    )

    report = OUT_ROOT / "report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("CONTROL VS GATE-ONLY — 3 SEEDS\n")
        f.write("=" * 72 + "\n\n")

        for row in mode_summary:
            f.write(
                f"{row['mode']:10s} "
                f"internal_acc={row['internal_accuracy_mean']:.4f}"
                f" ± {row['internal_accuracy_std']:.4f}  "
                f"internal_f1={row['internal_f1_mean']:.4f}"
                f" ± {row['internal_f1_std']:.4f}  "
                f"external_spec={row['external_specificity_mean']:.4f}"
                f" ± {row['external_specificity_std']:.4f}  "
                f"external_fpr={row['external_fpr_mean']:.4f}"
                f" ± {row['external_fpr_std']:.4f}\n"
            )

        f.write("\nPOR SPEAKER EXTERNO\n")
        f.write("-" * 72 + "\n")

        for speaker in speakers:
            f.write(f"\n{speaker}\n")
            for mode in MODES:
                row = next(
                    r for r in speaker_summary
                    if r["mode"] == mode and r["speaker_id"] == speaker
                )
                f.write(
                    f"  {mode:10s} "
                    f"spec={row['specificity_mean']:.4f}"
                    f" ± {row['specificity_std']:.4f}\n"
                )

    print()
    print("=" * 72)
    print("RESULTADO FINAL 3 SEEDS")
    print("=" * 72)

    for row in mode_summary:
        print(
            f"{row['mode']:10s} "
            f"Internal Acc={row['internal_accuracy_mean']:.4f}"
            f" ± {row['internal_accuracy_std']:.4f} | "
            f"External Spec={row['external_specificity_mean']:.4f}"
            f" ± {row['external_specificity_std']:.4f}"
        )

    print()
    print(f"Reporte:\n  {report}")


if __name__ == "__main__":
    main()
