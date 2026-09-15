import pandas as pd

p = r"C:\Users\gonza\TFG\Deepfake-Detection\experiments\baseline_logmel_cnn\test_predictions.csv"

df = pd.read_csv(p)

errores = df[df["label"] != df["prediction"]]

print(
    errores[
        [
            "path",
            "label",
            "prediction",
            "prob_fake",
            "generator",
            "speaker_id"
        ]
    ].to_string(index=False)
)