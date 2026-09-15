import pandas as pd

PATH = (
    r"C:\Users\gonza\TFG\Deepfake-Detection"
    r"\experiments\baseline_logmel_cnn"
    r"\test_predictions.csv"
)

df = pd.read_csv(PATH)

print("\nRESULTADOS POR GENERADOR\n")

for generator, group in df.groupby("generator"):

    y = group["label"]
    p = group["prediction"]

    total = len(group)
    correctos = (y == p).sum()

    accuracy = correctos / total

    tp = ((y == 1) & (p == 1)).sum()
    tn = ((y == 0) & (p == 0)).sum()
    fp = ((y == 0) & (p == 1)).sum()
    fn = ((y == 1) & (p == 0)).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(f"{generator}")
    print(f"  N         : {total}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print()