from pathlib import Path
import numpy as np
import soundfile as sf

ENTRADA = Path(r"C:\Users\gonza\TFG\Deepfake-Detection\data\real\WhatsApp Ptt 2026-09-10 at 17.38.21.ogg")
SALIDA = Path(r"C:\Users\gonza\TFG\Deepfake-Detection\data\real")

# Ajustables
SILENCIO_DB = -38
PAUSA_MIN = 0.35       # segundos
MARGEN = 0.10          # conservar algo alrededor de cada frase

SALIDA.mkdir(parents=True, exist_ok=True)

audio, sr = sf.read(
    str(ENTRADA),
    dtype="float32",
    always_2d=True
)

# Pasar a mono para detectar silencios
mono = audio.mean(axis=1)

# Amplitud correspondiente al umbral en dB
threshold = 10 ** (SILENCIO_DB / 20)

activo = np.abs(mono) > threshold

min_silence_samples = int(PAUSA_MIN * sr)

# Buscar bloques de voz separados por silencios largos
segments = []

start = None
silence_start = None

for i, is_active in enumerate(activo):

    if is_active:
        if start is None:
            start = i

        silence_start = None

    else:
        if start is not None:

            if silence_start is None:
                silence_start = i

            if i - silence_start >= min_silence_samples:

                end = silence_start

                segments.append((start, end))

                start = None
                silence_start = None

# Última frase
if start is not None:
    segments.append((start, len(audio)))

print(f"Segmentos detectados: {len(segments)}")

if len(segments) != 40:
    print(
        f"ATENCIÓN: esperaba 40 frases y he detectado {len(segments)}."
    )
    print(
        "Ajusta PAUSA_MIN o SILENCIO_DB."
    )

margin_samples = int(MARGEN * sr)

for n, (start, end) in enumerate(segments, 1):

    start = max(
        0,
        start - margin_samples
    )

    end = min(
        len(audio),
        end + margin_samples
    )

    fragment = audio[start:end]

    destino = SALIDA / f"Samu_{n:02d}.wav"

    sf.write(
        str(destino),
        fragment,
        sr,
        subtype="PCM_16"
    )

    duracion = len(fragment) / sr

    print(
        f"{n:03d}: "
        f"{duracion:.2f}s -> "
        f"{destino.name}"
    )