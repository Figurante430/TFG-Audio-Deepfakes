from __future__ import annotations

import argparse, csv, os, random, sys, traceback
from pathlib import Path

TFG = Path(r"C:\Users\gonza\TFG")
PROJECT = TFG / "Deepfake-Detection"
OUT_ROOT = PROJECT / "data" / "tts"
REF_DIR = PROJECT / "referencias_tts"
REF_TEXT_FILE = REF_DIR / "texto_referencia.txt"
PHRASES_FILE = PROJECT / "frases_200.txt"

N = 200
N_REFS = 5
PER_REF = 40
BASE_SEED = 20260908

REPOS = {
    "confucius4": [TFG / "Confucius4-TTS"],
    "cosyvoice": [TFG / "CosyVoice" / "CosyVoice"],
    "openVoice": [TFG / "OpenVoiceV2", TFG / "OpenVoice"],
    "qwen": [TFG / "Qwen3-TTS", TFG / "Qwen", TFG / "qwen3-tts"],
    "voxcpm2": [TFG / "VoxCPM", TFG / "VoxCPM2"],
}


def default_phrases():
    starts = [
        "Esta mañana", "Ayer por la tarde", "Al terminar la clase", "Durante el fin de semana",
        "Antes de salir de casa", "Después de comer", "Mientras esperaba el autobús",
        "Al llegar a la estación", "En una reunión tranquila", "Durante una conversación breve",
        "A primera hora del día", "Cuando terminó la lluvia", "En el centro de la ciudad",
        "Al volver del trabajo", "Durante las vacaciones", "Antes de comenzar la prueba",
        "En una tarde de septiembre", "Cuando sonó el teléfono", "Al abrir la ventana",
        "Después de revisar los datos",
    ]
    clauses = [
        "el equipo revisó los resultados antes de tomar una decisión definitiva.",
        "la biblioteca permaneció abierta hasta que terminó la última actividad.",
        "una persona preguntó por la dirección correcta y recibió una explicación clara.",
        "el ordenador guardó todos los documentos sin mostrar ningún mensaje de error.",
        "varios estudiantes comentaron las noticias y compararon diferentes puntos de vista.",
        "la cafetería preparó café, té y algunas tostadas para quienes acababan de llegar.",
        "el informe explicó los cambios principales con ejemplos sencillos y datos concretos.",
        "una llamada inesperada cambió los planes previstos para el resto de la jornada.",
        "el tren salió con puntualidad y llegó al destino pocos minutos antes de lo esperado.",
        "la conversación terminó con una pregunta breve sobre los próximos pasos del proyecto.",
    ]
    phrases = [f"{a}, {b}" for a in starts for b in clauses]
    assert len(phrases) == 200 and len(set(phrases)) == 200
    return phrases


def load_inputs():
    PROJECT.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    if not PHRASES_FILE.exists():
        PHRASES_FILE.write_text("\n".join(default_phrases()) + "\n", encoding="utf-8")
        print(f"[INFO] Creado {PHRASES_FILE}")

    phrases = [x.strip() for x in PHRASES_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(phrases) != N or len(set(phrases)) != N:
        raise RuntimeError(f"{PHRASES_FILE} debe tener exactamente 200 frases distintas.")

    refs = sorted(REF_DIR.glob("*.wav"), key=lambda p: p.name.lower())
    if len(refs) != N_REFS:
        raise RuntimeError(f"En {REF_DIR} debe haber exactamente 5 archivos .wav; hay {len(refs)}.")
    if not REF_TEXT_FILE.exists():
        raise RuntimeError(f"Falta {REF_TEXT_FILE}. Pon dentro la transcripción EXACTA de los 5 audios.")
    ref_text = REF_TEXT_FILE.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise RuntimeError("texto_referencia.txt está vacío.")
    return phrases, refs, ref_text


def repo_for(name):
    for p in REPOS[name]:
        if p.exists():
            return p
    raise RuntimeError(f"No encuentro el repo de {name}. Ajusta REPOS al principio del script.")


def enter_repo(repo):
    repo = Path(repo).resolve()

    os.chdir(repo)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def seed_all(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().squeeze().numpy()
    except Exception:
        pass
    import numpy as np
    return np.asarray(x).squeeze()


def qwen(repo, refs, ref_text):
    enter_repo(repo)
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    print("[Qwen] cargando modelo...")
    m = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="eager"
    )
    prompts = [m.create_voice_clone_prompt(str(r), ref_text, False) for r in refs]

    def gen(text, r, out, seed):
        seed_all(seed)
        wavs, sr = m.generate_voice_clone(
            text=text, language="Spanish", voice_clone_prompt=prompts[r], non_streaming_mode=True
        )
        sf.write(str(out), to_numpy(wavs[0]), sr)
    return gen


def cosyvoice(repo, refs, ref_text):

    # =========================================================
    # RUTA REAL DE COSYVOICE
    # =========================================================

    repo = Path(
        r"C:\Users\gonza\TFG\CosyVoice\CosyVoice"
    ).resolve()

    print("DEBUG repo:", repo)
    print("DEBUG repo existe:", repo.exists())
    print(
        "DEBUG cosyvoice.py existe:",
        (repo / "cosyvoice" / "cli" / "cosyvoice.py").exists()
    )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    matcha = repo / "third_party" / "Matcha-TTS"

    if matcha.exists() and str(matcha) not in sys.path:
        sys.path.insert(0, str(matcha))

    os.chdir(repo)

    print("DEBUG cwd:", os.getcwd())

    # =========================================================
    # IMPORTS
    # =========================================================

    import torch
    import torchaudio

    from cosyvoice.cli.cosyvoice import AutoModel

    print("CosyVoice importado OK")

    # =========================================================
    # LOCALIZAR COSYVOICE 3
    # =========================================================

    model_dir = (
        repo
        / "pretrained_models"
        / "Fun-CosyVoice3-0.5B"
    )

    # Si no está exactamente ahí, buscar SOLO cosyvoice3.yaml
    if not (model_dir / "cosyvoice3.yaml").exists():

        found = list(repo.glob("**/cosyvoice3.yaml"))

        if not found:
            raise RuntimeError(
                "No encuentro cosyvoice3.yaml. "
                "No parece estar descargado el modelo completo de CosyVoice3."
            )

        model_dir = found[0].parent

    print("[CosyVoice] Modelo encontrado:")
    print(model_dir)

    # Comprobar archivos necesarios
    required = [
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt"]

    faltan = [
        nombre
        for nombre in required
        if not (model_dir / nombre).exists()
    ]

    if faltan:
        raise RuntimeError(
            f"El modelo está incompleto. Faltan: {faltan}\n"
            f"Carpeta: {model_dir}"
        )

    print("[CosyVoice] Todos los archivos del modelo están presentes")

    m = AutoModel(
        model_dir=str(model_dir),
        fp16=False
    )

    print("[CosyVoice] modelo cargado correctamente")


    m = AutoModel(
        model_dir=str(model_dir),
        fp16=False
    )

    print("[CosyVoice] modelo cargado")

    # =========================================================
    # CARGAR MODELO
    # =========================================================

    m = AutoModel(
        model_dir=str(model_dir),
        fp16=False
    )

    print("[CosyVoice] modelo cargado")

    # =========================================================
    # REGISTRAR LAS 5 VOCES
    # =========================================================

    prompt = (
        "You are a helpful assistant."
        "<|endofprompt|>"
        + ref_text
    )

    spk = []

    print("[CosyVoice] preparando referencias...")

    for i, reference in enumerate(refs, 1):

        sid = f"dataset_ref_{i}"

        print(
            f"[CosyVoice] referencia {i}: "
            f"{reference.name}"
        )

        resultado = m.add_zero_shot_spk(
            prompt,
            str(reference),
            sid
        )

        if resultado is not True:
            raise RuntimeError(
                f"No se pudo registrar la referencia {i}"
            )

        spk.append(sid)

    print(
        f"[CosyVoice] {len(spk)} referencias preparadas"
    )

    # =========================================================
    # FUNCIÓN QUE GENERA UN AUDIO
    # =========================================================

    def gen(text, r, out, seed):

        seed_all(seed)

        chunks = []

        for resultado in m.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=spk[r],
            stream=False
        ):

            if "tts_speech" not in resultado:
                raise RuntimeError(
                    "CosyVoice no devolvió tts_speech"
                )

            chunks.append(
                resultado["tts_speech"]
                .detach()
                .cpu()
            )

        if not chunks:
            raise RuntimeError(
                "CosyVoice no devolvió audio"
            )

        audio = torch.cat(
            chunks,
            dim=-1
        )

        import soundfile as sf

        audio_np = (
            audio
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )

        sf.write(
            str(out),
            audio_np,
            m.sample_rate,
            subtype="PCM_16"
        )

    # ESTO ES CRÍTICO
    return gen


def confucius(repo, refs, ref_text):
    enter_repo(repo)
    import soundfile as sf
    import torch
    from confuciustts.cli.inference import ConfuciusTTS

    print("[Confucius] cargando modelo sin vLLM...")
    m = ConfuciusTTS(
        config_path=str(repo / "config" / "inference_config.yaml"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    def gen(text, r, out, seed):
        seed_all(seed)
        audio = m.generate(text=text, lang="es", prompt_wav=str(refs[r]), verbose=False)
        sf.write(str(out), to_numpy(audio), m.sample_rate)
    return gen


def voxcpm2(repo, refs, ref_text):
    enter_repo(repo)
    import soundfile as sf
    import torch, voxcpm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[VoxCPM2] cargando modelo...")
    m = voxcpm.VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        optimize=device.startswith("cuda"),
        device=device
    )

    def gen(text, r, out, seed):
        seed_all(seed)

        audio = m.generate(
            text=text,
            prompt_wav_path=str(refs[r]),
            prompt_text=ref_text,
            reference_wav_path=str(refs[r]),
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False,
        )

        sf.write(
            str(out),
            to_numpy(audio),
            m.tts_model.sample_rate
        )

    return gen


def openvoice(repo, refs, ref_text, out_dir):
    enter_repo(repo)

    import torch

    from melo.api import TTS
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # =========================================================
    # CHECKPOINTS OPENVOICE
    # =========================================================

    checkpoint_root = Path(
        r"C:\Users\gonza\TFG\OpenVoiceV2\OpenVoice\checkpoints_v2"
    )

    conv_dir = checkpoint_root / "converter"

    config_path = conv_dir / "config.json"
    checkpoint_path = conv_dir / "checkpoint.pth"

    if not config_path.exists():
        raise RuntimeError(
            f"No encuentro config.json del converter: {config_path}"
        )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No encuentro checkpoint.pth del converter: {checkpoint_path}"
        )

    # =========================================================
    # CARGAR TONE COLOR CONVERTER
    # =========================================================

    print("[OpenVoice] cargando converter...")

    conv = ToneColorConverter(
        str(config_path),
        device=device,
        enable_watermark=False
    )

    conv.load_ckpt(str(checkpoint_path))

    print("[OpenVoice] converter cargado OK")

    # =========================================================
    # EXTRAER EMBEDDINGS DE LAS 5 REFERENCIAS
    # =========================================================

    print("[OpenVoice] extrayendo embeddings de referencias...")

    targets = []

    for i, r in enumerate(refs, 1):
        print(f"[OpenVoice] referencia {i}: {r}")

        target_se, _ = se_extractor.get_se(
            str(r),
            conv,
            vad=False
        )

        targets.append(target_se)

    print(
        f"[OpenVoice] {len(targets)} referencias preparadas"
    )

    # =========================================================
    # CARGAR MELOTTS ESPAÑOL
    # =========================================================

    print("[OpenVoice] cargando MeloTTS español...")

    m = TTS(
        language="ES",
        device=device
    )

    speaker_ids = m.hps.data.spk2id

    print(
        "[OpenVoice] speakers MeloTTS:",
        speaker_ids
    )

    if "ES" not in speaker_ids:
        raise RuntimeError(
            f"MeloTTS no contiene speaker ES. "
            f"Disponibles: {speaker_ids}"
        )

    speaker_id = speaker_ids["ES"]

    print(
        "[OpenVoice] speaker español:",
        speaker_id
    )

    # =========================================================
    # EMBEDDING DE LA VOZ BASE ESPAÑOLA
    # =========================================================

    source_se_path = (
        checkpoint_root
        / "base_speakers"
        / "ses"
        / "es.pth"
    )

    if not source_se_path.exists():
        raise RuntimeError(
            f"No encuentro embedding base español: "
            f"{source_se_path}"
        )

    source_se = torch.load(
        str(source_se_path),
        map_location=device
    ).to(device)

    print(
        "[OpenVoice] embedding español cargado:",
        source_se_path
    )

    # =========================================================
    # TEMPORAL PARA SALIDA DE MELOTTS
    # =========================================================

    tmp = out_dir / "_tmp_openvoice.wav"

    # =========================================================
    # GENERACIÓN
    # =========================================================

    def gen(text, r, out, seed):
        seed_all(seed)

        # 1. MeloTTS genera español con su voz base
        m.tts_to_file(
            text,
            speaker_id,
            str(tmp),
            speed=1.0
        )

        # 2. OpenVoice transforma la voz base
        #    hacia la referencia correspondiente
        conv.convert(
            audio_src_path=str(tmp),
            src_se=source_se,
            tgt_se=targets[r],
            output_path=str(out),
            message="@MyShell"
        )

    return gen


def load_generator(name, repo, refs, ref_text, out_dir):
    if name == "qwen": return qwen(repo, refs, ref_text)
    if name == "cosyvoice": return cosyvoice(repo, refs, ref_text)
    if name == "confucius4": return confucius(repo, refs, ref_text)
    if name == "voxcpm2": return voxcpm2(repo, refs, ref_text)
    if name == "openVoice": return openvoice(repo, refs, ref_text, out_dir)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(REPOS))
    args = ap.parse_args()

    phrases, refs, ref_text = load_inputs()
    repo = repo_for(args.modelo)
    out_dir = OUT_ROOT / args.modelo
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {args.modelo}")
    print(f"Salida: {out_dir}")
    print("Distribución: 5 referencias x 40 frases = 200 audios")
    gen = load_generator(args.modelo, repo, refs, ref_text, out_dir)

    rows, errors = [], []
    for i, text in enumerate(phrases, 1):
        ref_idx = (i - 1) // PER_REF
        out = out_dir / f"{i:03d}_ref{ref_idx+1}.wav"
        seed = BASE_SEED + i
        status = "ok"

        if out.exists() and out.stat().st_size > 1000:
            print(f"[{i:03d}/200] SKIP {out.name}")
        else:
            try:
                print(f"[{i:03d}/200] ref{ref_idx+1} -> {out.name}")
                gen(text, ref_idx, out, seed)
                if not out.exists() or out.stat().st_size <= 1000:
                    raise RuntimeError("No se creó un WAV válido")
            except Exception as e:
                status = "error"
                errors.append(f"[{i:03d}] {type(e).__name__}: {e}\n{traceback.format_exc()}")
                print(f"ERROR [{i:03d}]: {e}")

        rows.append({
            "path": out.relative_to(PROJECT).as_posix(),
            "label": 1, "category": "tts", "generator": args.modelo,
            "reference_id": refs[ref_idx].stem, "reference_file": refs[ref_idx].name,
            "text_id": f"txt_{i:03d}", "text": text, "seed": seed, "status": status,
        })

    meta = out_dir / "metadata_generacion.csv"
    with meta.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    if errors:
        (out_dir / "errores_generacion.log").write_text("\n\n".join(errors), encoding="utf-8")

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nTerminado: {ok}/200 OK")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
