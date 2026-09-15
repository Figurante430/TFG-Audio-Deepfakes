from __future__ import annotations

import argparse
import csv
import inspect
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(r"C:\Users\gonza\TFG\Deepfake-Detection")
SOURCE_DIR = ROOT / "data" / "tts" / "confucius4"
STS_ROOT = ROOT / "data" / "sts"

KNNVC_REPO = Path(r"C:\Users\gonza\TFG\kNN-VC")
RVC_REPO = Path(r"C:\Users\gonza\TFG\Retrieval-based-Voice-Conversion-WebUI")
KNNVC_REF_DIR = ROOT / "referencias_tts"

POOL_SIZE = 250
N_OUTPUTS = 200
AUDIO_MIN_BYTES = 1000


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def collect_sources():
    wavs = sorted([p for p in SOURCE_DIR.glob("*.wav") if p.is_file()], key=natural_key)
    if len(wavs) < N_OUTPUTS:
        raise RuntimeError(f"Solo hay {len(wavs)} WAV en {SOURCE_DIR}; hacen falta {N_OUTPUTS}.")
    pool = wavs[:POOL_SIZE]
    selected = pool[:N_OUTPUTS]
    print(f"Fuente: {SOURCE_DIR}")
    print(f"WAV encontrados: {len(wavs)}")
    print(f"Pool: primeros {min(POOL_SIZE, len(wavs))}")
    print(f"Seleccionados: {len(selected)}")
    return selected


def valid_output(path: Path):
    return path.exists() and path.is_file() and path.stat().st_size > AUDIO_MIN_BYTES


def save_metadata(out_dir: Path, rows):
    csv_path = out_dir / "metadata_generacion.csv"
    fields = [
        "id", "source_path", "output_path", "category", "generator",
        "source_generator", "target", "status", "error"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Metadata: {csv_path}")


def prepare_16k_wav(src: Path, dst: Path):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    x = torch.from_numpy(audio).unsqueeze(0)
    if sr != 16000:
        x = AF.resample(x, sr, 16000)
    y = x.squeeze(0).cpu().numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, 16000, subtype="PCM_16")


def ref_number_from_source(src: Path, position: int):
    m = re.search(r"_ref([1-5])(?:_|$)", src.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return min(((position - 1) // 40) + 1, 5)


# =========================== kNN-VC ===========================

def run_knnvc(sources):
    if not KNNVC_REPO.exists():
        raise RuntimeError(f"No existe el repo kNN-VC: {KNNVC_REPO}")

    refs = sorted([p for p in KNNVC_REF_DIR.glob("*.wav") if p.is_file()], key=natural_key)
    if len(refs) != 5:
        raise RuntimeError(
            f"En {KNNVC_REF_DIR} deben existir exactamente 5 WAV. Ahora hay {len(refs)}."
        )

    out_dir = STS_ROOT / "knnvc"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = STS_ROOT / "_tmp_knnvc_16k"
    tmp_src_dir = tmp_dir / "sources"
    tmp_ref_dir = tmp_dir / "refs"
    tmp_src_dir.mkdir(parents=True, exist_ok=True)
    tmp_ref_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np

    # ============================================================
    # PARCHE: evitar TorchCodec al cargar WAV
    # ============================================================

    def safe_torchaudio_load(
        path,
        normalize=True,
        channels_first=True,
        format=None,
        buffer_size=4096,
        backend=None,
        **kwargs
    ):
        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True
        )

        # soundfile -> (samples, channels)
        # torchaudio -> (channels, samples)
        if channels_first:
            audio = audio.T

        audio = np.ascontiguousarray(audio)

        return torch.from_numpy(audio), sr


    # kNN-VC llama internamente a torchaudio.load().
    # Lo sustituimos por soundfile para no necesitar TorchCodec.
    torchaudio.load = safe_torchaudio_load

    print("[kNN-VC] torchaudio.load -> soundfile OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        trust_repo=True,
        prematched=True,
        pretrained=True,
        device=device,
    )

    print("[kNN-VC] preparando las 5 referencias a 16 kHz...")
    matching_sets = {}
    for ref_id, ref in enumerate(refs, 1):
        converted = tmp_ref_dir / f"ref{ref_id}.wav"
        if not valid_output(converted):
            prepare_16k_wav(ref, converted)
        print(f"[kNN-VC] matching set ref{ref_id}: {ref.name}")
        matching_sets[ref_id] = knn_vc.get_matching_set([str(converted)])

    rows = []
    ok = 0

    for i, src in enumerate(sources, 1):
        out = out_dir / src.name

        # Detecta de qué referencia procede el audio:
        # ref1, ref2, ref3, ref4 o ref5
        source_ref_id = ref_number_from_source(src, i)

        # Cambio circular:
        # ref1 -> ref2
        # ref2 -> ref3
        # ref3 -> ref4
        # ref4 -> ref5
        # ref5 -> ref1
        target_ref_id = (source_ref_id % 5) + 1

        target = refs[target_ref_id - 1]

        print(
            f"[{i:03d}/{len(sources)}] "
            f"ref{source_ref_id} -> ref{target_ref_id} "
            f"({refs[source_ref_id - 1].name} -> {target.name}) "
            f"-> {out.name}"
        )

        if valid_output(out):
            print("  SKIP: ya existe")
            ok += 1

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "knnvc",
                "source_generator": "confucius4",
                "target": str(target),
                "status": "skip",
                "error": ""
            })

            continue

        try:
            src16 = tmp_src_dir / f"{i:03d}_{src.name}"

            prepare_16k_wav(
                src,
                src16
            )

            # Características del audio original
            query_seq = knn_vc.get_features(
                str(src16)
            )

            # Conversión A LA SIGUIENTE VOZ
            out_wav = knn_vc.match(
                query_seq,
                matching_sets[target_ref_id],
                topk=4,
            )

            if hasattr(out_wav, "detach"):
                out_wav = (
                    out_wav
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            sf.write(
                str(out),
                out_wav,
                16000,
                subtype="PCM_16"
            )

            if not valid_output(out):
                raise RuntimeError(
                    "El WAV generado no es válido o está vacío"
                )

            ok += 1

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "knnvc",
                "source_generator": "confucius4",
                "target": str(target),
                "status": "ok",
                "error": ""
            })

        except Exception as e:
            print(
                f"  ERROR [{i:03d}]: {e}"
            )

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "knnvc",
                "source_generator": "confucius4",
                "target": str(target),
                "status": "error",
                "error": f"{type(e).__name__}: {e}"
            })

    save_metadata(out_dir, rows)
    print(f"\nTerminado kNN-VC: {ok}/{len(sources)} OK")


# ============================= RVC =============================

def discover_rvc_model(explicit: str | None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"No existe el modelo RVC: {p}")
        return p

    candidates = sorted((RVC_REPO / "assets" / "weights").glob("*.pth"), key=natural_key)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No encuentro ningún .pth en RVC\\assets\\weights. "
            "Ejecuta con --rvc-pth \"RUTA\\modelo.pth\"."
        )
    txt = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "Hay varios modelos .pth en assets\\weights. Indica cuál usar con --rvc-pth.\n" + txt
    )


def discover_rvc_index(model_path: Path, explicit: str | None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"No existe el índice RVC: {p}")
        return p

    all_indices = sorted(RVC_REPO.glob("logs/**/*added*.index"), key=natural_key)
    if not all_indices:
        return None

    stem = model_path.stem.lower()
    related = [p for p in all_indices if stem in str(p).lower()]
    if len(related) == 1:
        return related[0]
    if len(all_indices) == 1:
        return all_indices[0]
    return None


def enter_rvc_repo():
    repo = RVC_REPO.resolve()
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_rvc(model_path: Path):
    enter_rvc_repo()

    os.environ["weight_root"] = str(model_path.parent)
    os.environ.setdefault("index_root", str(RVC_REPO / "logs"))
    os.environ.setdefault("outside_index_root", str(RVC_REPO / "assets" / "indices"))
    os.environ.setdefault("rmvpe_root", str(RVC_REPO / "assets" / "rmvpe"))

    from configs.config import Config

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        config = Config()
    finally:
        sys.argv = old_argv

    try:
        from infer.modules.vc.modules import VC
    except ImportError:
        from infer.vc.modules import VC

    vc = VC(config)
    print(f"[RVC] cargando modelo: {model_path}")
    vc.get_vc(model_path.name)
    return vc


def call_rvc_single(vc, src: Path, index_path: Path | None):
    method = vc.vc_single
    params = inspect.signature(method).parameters
    idx = str(index_path) if index_path else ""
    index_rate = 0.75 if index_path else 0.0

    # API antigua
    if "f0_file" in params or "file_index2" in params:
        return method(
            0, str(src), 0, None, "rmvpe", idx, idx,
            index_rate, 3, 0, 1.0, 0.33,
        )

    # API actual
    return method(
        0, str(src), 0, "rmvpe", idx,
        index_rate, 0, 1.0, 0.33,
    )


def run_rvc(sources):
    import numpy as np
    import soundfile as sf

    # ============================================================
    # MODELO DESTINO: Mario_40
    # ============================================================

    model_path = (
        RVC_REPO
        / "assets"
        / "weights"
        / "Mario_40.pth"
    )

    if not model_path.exists():
        candidates = list(
            RVC_REPO.glob("**/Mario_40*.pth")
        )

        if not candidates:
            raise RuntimeError(
                "No encuentro el modelo Mario_40.pth dentro de:\n"
                f"{RVC_REPO}"
            )

        model_path = candidates[0]

    print()
    print("============================================")
    print("[RVC] MODELO DESTINO")
    print("============================================")
    print(f"[RVC] modelo: {model_path}")

    # ============================================================
    # BUSCAR ÍNDICE DE Mario_40
    # ============================================================

    index_candidates = [
        p
        for p in RVC_REPO.glob("**/*.index")
        if "mario_40" in str(p).lower()
    ]

    added_indices = [
        p
        for p in index_candidates
        if p.name.lower().startswith("added_")
    ]

    if added_indices:
        index_path = added_indices[0]
    elif index_candidates:
        index_path = index_candidates[0]
    else:
        index_path = None

    if index_path:
        print(f"[RVC] índice: {index_path}")
    else:
        print("[RVC] índice: NO ENCONTRADO")
        print("[RVC] se realizará inferencia sin índice")

    # ============================================================
    # CARPETA DE SALIDA
    # ============================================================

    out_dir = STS_ROOT / "rvc"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"[RVC] salida: {out_dir}")

    # ============================================================
    # CARGAR RVC
    # ============================================================

    print()
    print("[RVC] cargando Mario_40...")

    vc = load_rvc(
        model_path
    )

    print("[RVC] Mario_40 cargado OK")
    print()

    # ============================================================
    # GENERACIÓN
    # ============================================================

    rows = []
    ok = 0

    total = len(sources)

    for i, src in enumerate(sources, 1):

        out = out_dir / src.name

        print(
            f"[{i:03d}/{total}] "
            f"{src.name} -> Mario_40"
        )

        if valid_output(out):

            print("  SKIP: ya existe")

            ok += 1

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "rvc",
                "source_generator": "confucius4",
                "target": "Mario_40",
                "status": "skip",
                "error": ""
            })

            continue

        try:

            status, result = call_rvc_single(
                vc,
                src,
                index_path
            )

            if result is None:
                raise RuntimeError(
                    f"RVC no devolvió audio: {status}"
                )

            sample_rate, audio = result

            if hasattr(
                audio,
                "detach"
            ):
                audio = (
                    audio
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            audio = np.asarray(audio)
            audio = np.squeeze(audio)

            if audio.size == 0:
                raise RuntimeError(
                    "RVC devolvió un audio vacío"
                )

            sf.write(
                str(out),
                audio,
                int(sample_rate),
                subtype="PCM_16"
            )

            if not valid_output(out):
                raise RuntimeError(
                    "El WAV generado no es válido o está vacío"
                )

            ok += 1

            print(
                f"  OK -> {out.name}"
            )

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "rvc",
                "source_generator": "confucius4",
                "target": "Mario_40",
                "status": "ok",
                "error": ""
            })

        except Exception as e:

            print(
                f"  ERROR [{i:03d}]: "
                f"{type(e).__name__}: {e}"
            )

            rows.append({
                "id": i,
                "source_path": str(src),
                "output_path": str(out),
                "category": "sts",
                "generator": "rvc",
                "source_generator": "confucius4",
                "target": "Mario_40",
                "status": "error",
                "error": (
                    f"{type(e).__name__}: {e}"
                )
            })

    # ============================================================
    # METADATA
    # ============================================================

    save_metadata(
        out_dir,
        rows
    )

    # ============================================================
    # RESUMEN
    # ============================================================

    print()
    print("============================================")
    print("RVC TERMINADO")
    print("============================================")

    print(
        f"Correctos: {ok}/{total}"
    )

    print(
        f"Salida: {out_dir}"
    )

    print(
        "Voz destino: Mario_40"
    )


# ============================= MAIN =============================

def main():
    parser = argparse.ArgumentParser(
        description="Genera el dataset STS (kNN-VC / RVC) a partir de Confucius4."
    )
    parser.add_argument("--modelo", required=True, choices=["knnvc", "rvc"])
    parser.add_argument("--rvc-pth", help="Ruta al .pth de RVC. Solo necesaria si hay varios modelos.")
    parser.add_argument("--rvc-index", help="Ruta al added_*.index de RVC. Opcional.")
    args = parser.parse_args()

    sources = collect_sources()
    print(f"Modelo: {args.modelo}")
    print(f"Salida: {STS_ROOT / args.modelo}")
    print(f"Distribución: {len(sources)} audios")

    if args.modelo == "knnvc":
        run_knnvc(sources)
    else:
        run_rvc(sources)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
