# Gestos-Spotify

Control de Spotify mediante gestos de mano usando webcam, visión por computadora y Machine Learning.

## Descripción

Este proyecto permite controlar Spotify (Play/Pause, Siguiente/Anterior, Subir/Bajar volumen) mediante gestos de mano capturados por webcam. Utiliza extracción de características HOG + clasificador RandomForest para reconocer 6 gestos distintos.

## Gestos soportados

| Gesto | Acción Spotify | Tecla multimedia |
|-------|----------------|------------------|
| `play_stop` | Play / Pause | `Play/Pause` |
| `siguienteCancion` | Siguiente canción | `Next Track` |
| `anteriorCancion` | Canción anterior | `Previous Track` |
| `subirVolumen` | Subir volumen | `Volume Up` (x5) |
| `bajarVolumen` | Bajar volumen | `Volume Down` (x5) |
| `up` | Play / Pause (alternativo) | `Play/Pause` |

## Requisitos

- Python 3.8+
- Webcam
- Spotify abierto (en segundo plano funciona en Windows)

## Instalación

```bash
git clone https://github.com/tu-usuario/Gestos-Spotify.git
cd Gestos-Spotify
pip install -r requirements.txt
```

## Uso

```bash
python detector_gestos.py
```

Controles:
- Coloca la mano dentro del rectángulo verde
- Presiona **Q** para salir

## Estructura del proyecto

```
Gestos-Spotify/
├── detector_gestos.py          # Aplicación principal
├── requirements.txt            # Dependencias Python
├── .gitignore                  # Archivos ignorados por Git
├── dataset-kaggle2/            # Dataset principal (5 clases, ~2.6k imágenes)
│   └── images/
│       ├── anteriorCancion/
│       ├── bajarVolumen/
│       ├── play_stop/
│       ├── siguienteCancion/
│       └── subirVolumen/
├── dataset-Manual/             # Dataset manual capturado (6 clases, ~150 imágenes)
│   ├── anteriorCancion/
│   ├── bajarVolumen/
│   ├── nada/
│   ├── play_stop/
│   ├── siguienteCancion/
│   └── subirVolumen/
└── scripts/                    # Utilidades de preprocesamiento
    ├── redimensionar_dataset.py    # Redimensionar imágenes a 64x64
    ├── scriptRotar180.py           # Rotar 180° (data augmentation)
    ├── scriptRotar90.py            # Rotar 90° (data augmentation)
    └── scriptVoltear.py            # Voltear horizontal (data augmentation)
```

## Pipeline técnico

1. **Preprocesamiento**: BGR → Gris → Resize 64×64 → Gaussian Blur → Otsu Threshold → Normalización
2. **Extracción de características**: HOG (Histogram of Oriented Gradients)
   - 9 orientaciones, cells 8×8, blocks 2×2, L2-Hys normalization
3. **Clasificación**: RandomForest (200 árboles, max_depth=None, min_samples_leaf=2)
4. **Detección tiempo real**: Suavizado temporal (6 frames) + umbral de confianza (35%)
5. **Control Spotify**: pyautogui con teclas multimedia (cooldown 1s por gesto)

## Reentrenamiento

El modelo se reentrena automáticamente si no existe `modelo_gestos.pkl` o si el usuario confirma. Para forzar reentrenamiento:

```bash
rm modelo_gestos.pkl
python detector_gestos.py
```

## Scripts de utilidad

```bash
# Redimensionar dataset
python scripts/redimensionar_dataset.py dataset-kaggle2

# Data augmentation (ejemplos hardcodeados en cada script)
python scripts/scriptRotar180.py
python scripts/scriptRotar90.py
python scripts/scriptVoltear.py
```

## Licencia

MIT