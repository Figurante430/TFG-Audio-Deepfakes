import os
import time
from pathlib import Path

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ReadTimeout,
)

DATASET_ID = "cmqim2spa00synr071fcp7av0"

OUTPUT = Path(
    r"C:\Users\gonza\TFG\Deepfake-Detection"
    r"\common_voice_es_26.tar.gz"
)

API_URL = (
    "https://mozilladatacollective.com/api/datasets/"
    f"{DATASET_ID}/download"
)

api_key = os.environ["MDC_API_KEY"]

MAX_RETRIES = 20
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def get_download_url():
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["downloadUrl"]


def descargar():
    for intento in range(1, MAX_RETRIES + 1):

        descargado = (
            OUTPUT.stat().st_size
            if OUTPUT.exists()
            else 0
        )

        print()
        print(
            f"Intento {intento}/{MAX_RETRIES}"
        )

        print(
            "Ya descargado:",
            f"{descargado / 1024**3:.2f} GB"
        )

        # Pedimos una URL temporal nueva en cada intento
        download_url = get_download_url()

        headers = {}

        if descargado > 0:
            headers["Range"] = (
                f"bytes={descargado}-"
            )

        try:
            with requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as r:

                # Si pedimos Range queremos 206 Partial Content
                if descargado > 0:

                    if r.status_code == 206:
                        modo = "ab"

                    elif r.status_code == 200:
                        raise RuntimeError(
                            "El servidor no aceptó la "
                            "reanudación mediante Range. "
                            "No se sobrescribirá el archivo."
                        )

                    else:
                        r.raise_for_status()

                else:
                    r.raise_for_status()
                    modo = "wb"

                content_length = int(
                    r.headers.get(
                        "content-length",
                        0
                    )
                )

                total = (
                    descargado + content_length
                    if content_length
                    else 0
                )

                with OUTPUT.open(modo) as f:

                    for chunk in r.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        f.write(chunk)

                        descargado += len(chunk)

                        if total:
                            porcentaje = (
                                descargado
                                / total
                                * 100
                            )

                            print(
                                "\r"
                                f"Descargando: "
                                f"{porcentaje:6.2f}% "
                                f"("
                                f"{descargado / 1024**3:.2f}"
                                f"/"
                                f"{total / 1024**3:.2f}"
                                f" GB)",
                                end="",
                            )

                        else:
                            print(
                                "\r"
                                f"Descargados: "
                                f"{descargado / 1024**3:.2f}"
                                f" GB",
                                end="",
                            )

                print()
                print("Descarga terminada.")
                print(OUTPUT)

                return

        except (
            ChunkedEncodingError,
            ConnectionError,
            ReadTimeout,
        ) as e:

            print()
            print(
                "La conexión se ha cortado:"
            )
            print(type(e).__name__)

            print(
                "Esperando 10 segundos y "
                "reanudando..."
            )

            time.sleep(10)

    raise RuntimeError(
        "Se agotó el número máximo de reintentos."
    )


if __name__ == "__main__":
    descargar()