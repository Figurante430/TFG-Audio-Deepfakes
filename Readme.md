# Generación y detección de deepfakes de audio

Repositorio asociado a un Trabajo Fin de Grado centrado en la **generación, evaluación y detección de deepfakes de audio**, con especial atención a la **generalización fuera de dominio** de los detectores.

El proyecto estudia dos bloques principales:

- **Generación de voz sintética y clonada** mediante varios sistemas de TTS, voice cloning y voice conversion.
- **Detección de deepfakes de audio** mediante un baseline CNN sobre espectrogramas Log-Mel y modelos basados en **WavLM**.

El objetivo no es únicamente obtener buenas métricas en un conjunto interno, sino comprobar hasta qué punto los detectores mantienen su rendimiento ante **voces reales externas** y **métodos de generación diferentes**.

---

## Objetivos

Los objetivos principales del proyecto son:

1. Construir un conjunto de datos con audio real y audio sintético generado mediante distintas familias de modelos.
2. Entrenar detectores de deepfakes de audio con arquitecturas y estrategias diferentes.
3. Analizar la generalización entre generadores mediante experimentos LOGO (*Leave-One-Generator-Out*).
4. Analizar la generalización entre hablantes mediante experimentos LOSO (*Leave-One-Speaker-Out*).
5. Estudiar posibles *shortcuts* acústicos y sesgos de dominio.
6. Evaluar distintas estrategias de robustez y adaptación de dominio.
7. Realizar una evaluación externa final sobre un holdout congelado y no utilizado para reajustar los modelos.

---

## Sistemas de generación estudiados

El trabajo utiliza muestras procedentes de varios sistemas de síntesis y conversión de voz:

- **RVC**
- **Qwen3-TTS**
- **Fun-CosyVoice3**
- **Confucius4-TTS**
- **OpenVoice V2**
- **VoxCPM2**
- **KNN-VC** en experimentos internos

Los repositorios y pesos originales de estos sistemas son dependencias externas y **no se incluyen en este repositorio**.

---

## Detectores implementados

### CNN Log-Mel

Baseline basado en espectrogramas Log-Mel y una red neuronal convolucional.

Se utiliza como referencia frente a los modelos basados en representaciones preentrenadas.

### WavLM

Los experimentos principales utilizan `microsoft/wavlm-base-plus` como extractor de representaciones acústicas.

Sobre esta base se han probado diferentes estrategias:

- **WavLM inicial**
- **WavLM Control**
- **WavLM Robust**
- **WavLM Gate-only**
- **WavLM Channel-mix**
- **WavLM Domain-adapt real**

Las variantes robustas introducen transformaciones acústicas durante el entrenamiento, mientras que la adaptación de dominio incorpora voz real externa únicamente en entrenamiento para reducir la dependencia del modelo respecto al dominio de la clase real.

---

## Estructura del repositorio

```text
TFG/
├── .gitignore
├── Deepfake-Detection/
│   ├── Scripts/
│   │   ├── 01_data_preparation/
│   │   ├── 02_generation/
│   │   ├── 03_freeze_manifests/
│   │   ├── 04_training/
│   │   ├── 05_evaluation/
│   │   ├── 06_analysis/
│   │   └── 07_common_voice/
│   │
│   ├── data/
│   ├── data_normalized/
│   ├── experiments/
│   ├── external_final_fake/
│   ├── external_final_holdout/
│   ├── external_final_voxpopuli/
│   ├── external_results/
│   ├── external_test/
│   ├── referencias_tts/
│   ├── resources/
│   └── requirements.txt
│
└── README.md
```

### `Scripts/01_data_preparation`

Preparación, normalización y validación del dataset.

Incluye, entre otros:

- creación de metadatos;
- creación de particiones train/validation/test;
- normalización del conjunto;
- validación de archivos;
- preparación de VoxPopuli para evaluación externa.

### `Scripts/02_generation`

Scripts utilizados para preparar trabajos y generar muestras externas con los distintos sistemas de síntesis y clonación.

### `Scripts/03_freeze_manifests`

Scripts destinados a congelar los conjuntos de evaluación, generar manifiestos SHA-256 y construir los resúmenes finales.

### `Scripts/04_training`

Scripts de entrenamiento de los detectores CNN y WavLM, incluyendo experimentos LOGO, LOSO y variantes de robustez/adaptación de dominio.

### `Scripts/05_evaluation`

Evaluación sobre voz real externa y sobre el holdout final.

### `Scripts/06_analysis`

Análisis de *shortcuts* acústicos, comparaciones entre variantes y métricas por generador.

### `Scripts/07_common_voice`

Descarga y selección de subconjuntos de Common Voice utilizados en adaptación y evaluación externa.

### `experiments`

Resultados de los experimentos internos, incluyendo métricas, historiales y predicciones.

### `external_results`

Resultados de la evaluación externa final.

### `external_final_holdout`

Metadatos y manifiesto del conjunto externo final congelado.

---

## Dataset y protocolo experimental

El proyecto utiliza audio real y sintético procedente de diferentes fuentes.

Los archivos de audio **no se incluyen en GitHub** debido a su tamaño y a cuestiones de redistribución. El repositorio conserva los metadatos, manifiestos, hashes, resultados y scripts necesarios para documentar el proceso experimental.

La evaluación externa final utiliza un holdout de:

- **220 audios** en total
- **100 audios reales** procedentes de VoxPopuli
- **120 audios falsos**
- **6 familias de generación**
- **20 audios por generador**

Generadores incluidos en el holdout final:

- RVC
- Qwen3-TTS
- Fun-CosyVoice3
- Confucius4-TTS
- OpenVoice V2
- VoxCPM2

### Huella del holdout final

```text
3fed1b260a404af75b21aab8a57f1cf2dd0dbc8967d2695b45780095008f6021
```

El conjunto fue congelado antes de la evaluación final y no se modificó posteriormente.

Los thresholds de decisión se calibraron utilizando datos de validación y **no se reajustaron con el holdout externo**.

---

## Resultados finales

| Detector | Balanced Accuracy | Recall fake | Specificity real | F1 | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|
| CNN Log-Mel | 0.5725 | 0.6750 | 0.4700 | 0.6378 | 0.5899 | 0.4592 |
| WavLM inicial | 0.5917 | 0.8333 | 0.3500 | 0.7018 | 0.6008 | 0.4592 |
| WavLM Control | 0.6519 ± 0.0509 | 0.8472 ± 0.0376 | 0.4567 ± 0.1387 | **0.7375 ± 0.0177** | **0.7409 ± 0.0239** | **0.3061 ± 0.0106** |
| WavLM Robust | 0.6103 ± 0.0255 | **0.8639 ± 0.0481** | 0.3567 ± 0.0987 | 0.7199 ± 0.0027 | 0.6756 ± 0.0268 | 0.3544 ± 0.0253 |
| WavLM Gate-only | 0.6436 ± 0.0351 | 0.8306 ± 0.0459 | 0.4567 ± 0.1159 | 0.7277 ± 0.0058 | 0.7197 ± 0.0324 | 0.3078 ± 0.0172 |
| WavLM Channel-mix | 0.6344 ± 0.0113 | 0.8556 ± 0.0268 | 0.4133 ± 0.0493 | 0.7298 ± 0.0019 | 0.6864 ± 0.0086 | 0.3531 ± 0.0140 |
| WavLM Domain-adapt real | **0.6608 ± 0.0128** | 0.7917 ± 0.0520 | **0.5300 ± 0.0265** | 0.7247 ± 0.0228 | 0.7235 ± 0.0053 | 0.3317 ± 0.0000 |

### Lectura principal de los resultados

- **WavLM Control** obtiene la mejor capacidad discriminativa global, con el mayor ROC-AUC medio y el menor EER.
- **WavLM Domain-adapt real** obtiene la mejor balanced accuracy y la mayor especificidad sobre voz real.
- **WavLM Robust** consigue el mayor recall medio de deepfakes, aunque incrementa los falsos positivos sobre voz real.
- La adaptación con voz real externa mejora la generalización frente al cambio de dominio.
- Las augmentations acústicas genéricas no mejoran de forma sistemática el rendimiento global.
- **RVC** aparece como el generador más difícil de detectar de forma consistente.
- Confucius4-TTS, OpenVoice V2 y VoxCPM2 presentan tasas de detección considerablemente más altas en las variantes WavLM.

Estas comparaciones son descriptivas. No se seleccionó posteriormente una semilla concreta ni se recalibraron los modelos usando el holdout final.

---

## Fingerprint de resultados finales

El paquete final de resultados fue congelado con la siguiente huella SHA-256:

```text
f2f4d960cd9a41382e9797481cd8a17f6f69c4fa29d6dadd57de318c894171c4
```

Los ficheros correspondientes se encuentran en:

```text
Deepfake-Detection/external_results/FINAL_DETECTION_RESULTS/
```

Incluyen:

- `detector_summary.csv`
- `generator_summary.csv`
- `source_files_sha256.csv`
- `tables_for_tfg.md`
- `final_detection_report.json`
- `final_results_manifest.json`

---

## Instalación

Se recomienda utilizar un entorno virtual de Python.

```powershell
cd Deepfake-Detection
python -m venv detector-env
.\detector-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Algunas dependencias, especialmente PyTorch, pueden requerir una instalación específica según la versión de CUDA y la GPU utilizada.

---

## Ejecución

Después de la reorganización del proyecto, los scripts se ejecutan desde la raíz de `Deepfake-Detection`.

### Ejemplo: entrenamiento del baseline CNN

```powershell
python Scripts\04_training\entrenar_baseline_cnn.py
```

### Ejemplo: entrenamiento WavLM Control

```powershell
python Scripts\04_training\entrenar_wavlm_domain_robust.py --mode control --seed 42
```

### Ejemplo: entrenamiento WavLM Robust

```powershell
python Scripts\04_training\entrenar_wavlm_domain_robust.py --mode robust --seed 42
```

### Ejemplo: evaluación final CNN

```powershell
python Scripts\05_evaluation\evaluate_external_final_cnn.py
```

### Ejemplo: evaluación final WavLM

```powershell
python Scripts\05_evaluation\evaluate_external_final_wavlm.py
```

### Ejemplo: evaluación de la suite robusta

```powershell
python Scripts\05_evaluation\evaluate_external_final_wavlm_robust_suite.py
```

### Construcción del resumen final

```powershell
python Scripts\03_freeze_manifests\build_final_detection_summary.py
```

---

## Reproducibilidad

El repositorio conserva:

- scripts de preparación de datos;
- scripts de generación;
- configuración experimental;
- semillas utilizadas;
- metadatos;
- manifiestos de congelación;
- hashes SHA-256;
- métricas;
- predicciones;
- resultados por generador;
- resultados por semilla;
- resúmenes finales.

Las variantes WavLM principales fueron evaluadas con las semillas:

```text
42
123
2026
```

Los resultados multi-semilla se presentan como **media ± desviación estándar**.

---

## Archivos no incluidos

Para mantener el repositorio ligero y evitar redistribuir contenido de terceros, el `.gitignore` excluye:

- audios (`.wav`, `.flac`, `.mp3`, etc.);
- entornos virtuales;
- checkpoints y pesos de modelos;
- caches de Hugging Face y PyTorch;
- archivos comprimidos y datasets binarios pesados;
- repositorios externos utilizados para síntesis y clonación.

Entre los repositorios externos no incluidos se encuentran los correspondientes a RVC, Qwen3-TTS, CosyVoice, Confucius4-TTS, OpenVoice V2 y VoxCPM2.

---

## Consideraciones metodológicas

La evaluación externa se diseñó para evitar una estimación excesivamente optimista del rendimiento.

En particular:

- el holdout externo se congeló antes de la evaluación final;
- no se modificaron muestras después del congelado;
- los thresholds proceden de validación previa;
- el EER calculado sobre el holdout se utiliza únicamente como métrica diagnóstica;
- no se seleccionó la mejor semilla después de observar el holdout;
- no se continuó ajustando el detector a partir de los resultados del conjunto final.

El objetivo es medir capacidad de **generalización**, no optimizar específicamente el rendimiento sobre un único test.

---

## Limitaciones

Los resultados muestran que la detección de deepfakes de audio sigue siendo altamente dependiente del dominio y del método de generación.

Entre las principales limitaciones se encuentran:

- tamaño limitado del holdout externo;
- diferencias acústicas entre datasets;
- fuerte variabilidad entre generadores;
- sensibilidad del detector a la distribución de voz real;
- dificultad particular de los sistemas de voice conversion como RVC;
- ausencia de evaluación forense en condiciones reales de producción.

Los modelos desarrollados deben entenderse como **prototipos experimentales de investigación** y no como sistemas forenses listos para producción.

---

## Uso académico

Este repositorio forma parte de un Trabajo Fin de Grado sobre **generación y detección de deepfakes de audio**.

El código y los resultados se publican con fines educativos, de investigación y reproducibilidad experimental.

Los modelos generativos externos mantienen sus respectivas licencias y condiciones de uso.
