# Resultados finales de detección

Holdout externo: **220 audios (100 reales + 120 falsos)**.

Fingerprint SHA-256: `3fed1b260a404af75b21aab8a57f1cf2dd0dbc8967d2695b45780095008f6021`

> Los modelos con tres semillas se > expresan como media ± desviación estándar. > CNN y WavLM inicial son ejecuciones únicas.

## Tabla global

| Detector | N | Accuracy | Balanced Acc. | Recall fake | Specificity real | F1 | ROC-AUC | EER | Macro recall fake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN Log-Mel | 1 | 0.5818 | 0.5725 | 0.6750 | 0.4700 | 0.6378 | 0.5899 | 0.4592 | 0.6750 |
| WavLM inicial | 1 | 0.6136 | 0.5917 | 0.8333 | 0.3500 | 0.7018 | 0.6008 | 0.4592 | 0.8333 |
| WavLM Control | 3 | 0.6697 ± 0.0430 | 0.6519 ± 0.0509 | 0.8472 ± 0.0376 | 0.4567 ± 0.1387 | 0.7375 ± 0.0177 | 0.7409 ± 0.0239 | 0.3061 ± 0.0106 | 0.8472 ± 0.0376 |
| WavLM Robust | 3 | 0.6333 ± 0.0189 | 0.6103 ± 0.0255 | 0.8639 ± 0.0481 | 0.3567 ± 0.0987 | 0.7199 ± 0.0027 | 0.6756 ± 0.0268 | 0.3544 ± 0.0253 | 0.8639 ± 0.0481 |
| WavLM Gate-only | 3 | 0.6606 ± 0.0278 | 0.6436 ± 0.0351 | 0.8306 ± 0.0459 | 0.4567 ± 0.1159 | 0.7277 ± 0.0058 | 0.7197 ± 0.0324 | 0.3078 ± 0.0172 | 0.8306 ± 0.0459 |
| WavLM Channel-mix | 3 | 0.6545 ± 0.0079 | 0.6344 ± 0.0113 | 0.8556 ± 0.0268 | 0.4133 ± 0.0493 | 0.7298 ± 0.0019 | 0.6864 ± 0.0086 | 0.3531 ± 0.0140 | 0.8556 ± 0.0268 |
| WavLM Domain-adapt real | 3 | 0.6727 ± 0.0164 | 0.6608 ± 0.0128 | 0.7917 ± 0.0520 | 0.5300 ± 0.0265 | 0.7247 ± 0.0228 | 0.7235 ± 0.0053 | 0.3317 ± 0.0000 | 0.7917 ± 0.0520 |

## Resultados por generador

| Origen | Métrica | CNN Log-Mel | WavLM inicial | WavLM Control | WavLM Robust | WavLM Gate-only | WavLM Channel-mix | WavLM Domain-adapt real |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VoxPopuli | Specificity | 0.4700 | 0.3500 | 0.4567 ± 0.1387 | 0.3567 ± 0.0987 | 0.4567 ± 0.1159 | 0.4133 ± 0.0493 | 0.5300 ± 0.0265 |
| RVC | Fake recall | 0.6500 | 0.6000 | 0.4000 ± 0.0866 | 0.6000 ± 0.1323 | 0.4500 ± 0.1323 | 0.5500 ± 0.0000 | 0.3833 ± 0.1041 |
| Qwen3-TTS | Fake recall | 0.6000 | 0.8500 | 0.9167 ± 0.0577 | 0.9000 ± 0.0500 | 0.8500 ± 0.0500 | 0.8667 ± 0.0289 | 0.8667 ± 0.0577 |
| Fun-CosyVoice3 | Fake recall | 0.6000 | 0.8000 | 0.8000 ± 0.0500 | 0.7500 ± 0.0500 | 0.7333 ± 0.0289 | 0.7833 ± 0.0289 | 0.7333 ± 0.0764 |
| Confucius4-TTS | Fake recall | 0.6500 | 1.0000 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9000 ± 0.0866 |
| OpenVoice V2 | Fake recall | 0.9000 | 1.0000 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 | 0.9833 ± 0.0289 |
| VoxCPM2 | Fake recall | 0.6500 | 0.7500 | 1.0000 ± 0.0000 | 0.9667 ± 0.0289 | 0.9833 ± 0.0289 | 0.9667 ± 0.0577 | 0.8833 ± 0.0577 |

## Nota metodológica

Los umbrales de decisión fueron establecidos antes de la evaluación del holdout externo. No se recalibró ningún detector utilizando los 220 audios del conjunto final.

Las clasificaciones relativas que puedan derivarse de estas tablas son descriptivas y no deben utilizarse para entrenar, recalibrar o seleccionar nuevas variantes empleando este mismo holdout.
