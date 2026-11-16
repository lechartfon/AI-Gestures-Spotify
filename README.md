# AI Gestures Spotify 

Control Spotify using hand gestures detected by webcam using Computer Vision + Machine Learning.

## Demo

| Gesture | Action |
|---------|--------|
| ✋ Palm | Play / Pause |
| 👉 Next | Next song |
| 👈 Before | Previous song |
| 👍 Good | Volume up |
| 👎 Bad | Volume down |

## Requirements

- Python 3.8+
- Webcam
- Windows (uses Windows multimedia keys)
- Spotify open (in foreground or background)

## Installation

```bash
git clone https://github.com/lechartfon/AI-Gestures-Spotify.git
cd AI-Gestures-Spotify
pip install -r requirements.txt
```

## Usage

```bash
python gesture_detector.py
```

1. The first time, train the model (takes ~30-60s depending on CPU).
2. The webcam window opens with a green rectangle.
3. Place your hand inside the rectangle.
4. Perform the gestures shown in the table above.
5. Press **Q** to exit.

## Project Structure

```
AI-Gestures-Spotify/
├── gesture_detector.py     # Complete pipeline: preprocessing, training, inference
├── utils/
│   └── dataset_tools.py    # Utilities: resize, data augmentation
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The dataset and the trained model (`gesture_model.pkl`) are in `.gitignore` due to their size. Download the dataset from Kaggle or capture your own images.

## Dataset

The code automatically detects the folder structure:

```
dataset-kaggle2/           # Kaggle structure (with images/ subfolder)
└── images/
    ├── previous_song/
    ├── volume_down/
    ├── play_stop/
    ├── next_song/
    └── volume_up/
```

## Data Preparation (`utils/dataset_tools.py`)

```bash
# Resize entire dataset to 64x64
python utils/dataset_tools.py resize dataset-kaggle2

# Data augmentation: rotate 180° and flip
python utils/dataset_tools.py augment dataset-kaggle2/images/next_song dataset-kaggle2/images/previous_song --rotate 180 --flip
```

**Available commands:**
- `resize <source> [destination] [width] [height]` — Resizes all images
- `augment <source> <destination> --rotate 90|180|270 --flip` — Increases dataset with rotations/flip

## Technical Pipeline

| Stage | Description |
|-------|-------------|
| **Preprocessing** | BGR → Gray → Resize 64×64 → Gaussian Blur → Otsu → Normalize → HOG (9 ori, 8×8 cells, 2×2 blocks) |
| **Model** | RandomForestClassifier (200 trees, max_depth=None, min_samples_leaf=2) |
| **Evaluation** | 80/20 stratified split, confusion matrix, classification report |
| **Inference** | Webcam → ROI → Same pipeline → Predict_proba → Temporal smoothing (6 frames) → Cooldown 1s |

## Customization

Edit constants at the beginning of `gesture_detector.py`:

```python
DATASET_PATH = "dataset-kaggle2/"   # Path to dataset
MODEL_PATH   = "gesture_model.pkl"  # Model file
IMG_SIZE     = (64, 64)             # Input resolution
COOLDOWN     = 1.0                  # Seconds between repeated gestures
THRESHOLD_FRAMES = 6                # Consecutive frames to confirm
MIN_CONFIDENCE = 0.35               # Minimum probability threshold
```

## Main dependencies

| Package | Use |
|---------|-----|
| `opencv-python` | Webcam capture, image processing |
| `scikit-image` | HOG features |
| `scikit-learn` | RandomForest, metrics |
| `joblib` | Model serialization |
| `pyautogui` | Simulate multimedia keys |
| `matplotlib` | Confusion matrix visualization |

## Troubleshooting

| Problem | Solution |
|----------|----------|
| Webcam won't open | Check camera index (`cv2.VideoCapture(0)` → `1`, `2`...) |
| Gestures not detected | Improve lighting, uniform background, hand inside rectangle |
| Spotify won't respond | Ensure Spotify is open and accepts multimedia keys |
| Low model accuracy | Add more images to dataset, retrain by deleting `gesture_model.pkl` |

## License

MIT License - Free for personal and commercial use.