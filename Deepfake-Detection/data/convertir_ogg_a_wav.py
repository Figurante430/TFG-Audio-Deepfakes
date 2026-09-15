from pathlib import Path
import subprocess

REAL_DIR = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection\data\real"
)

ogg_files = list(REAL_DIR.rglob("*.ogg"))

print(f"OGG encontrados: {len(ogg_files)}")

for i, src in enumerate(ogg_files, 1):

    dst = src.with_suffix(".wav")

    print(
        f"[{i:03d}/{len(ogg_files)}] "
        f"{src.name} -> {dst.name}"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(src),

            # Mono
            "-ac", "1",

            # 16 kHz
            "-ar", "16000",

            # PCM16
            "-c:a", "pcm_s16le",

            str(dst)
        ],
        check=True
    )

print("Conversión terminada.")