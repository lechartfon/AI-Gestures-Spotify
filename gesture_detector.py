import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import pyautogui
import time
import matplotlib.pyplot as plt

# pyautogui: disable failsafe so it doesn't interrupt when moving mouse to corner
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # no pause between calls for maximum speed

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH = "dataset-kaggle2/"
MODEL_PATH   = "gesture_model.pkl"
IMG_SIZE     = (64, 64)

# HOG parameters (same for both dataset and webcam)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
)

GESTURE_LABELS = {
    "play_stop":        "Play / Stop",
    "volume_up":        "Volume Up",
    "volume_down":      "Volume Down",
    "next_song":        "Next Song",
    "previous_song":    "Previous Song",
}

# Spotify multimedia keys (work with Spotify in background on Windows)
ACTIONS = {
    "play_stop":        lambda: pyautogui.press('playpause'),
    "next_song":        lambda: pyautogui.press('nexttrack'),
    "previous_song":    lambda: pyautogui.press('prevtrack'),
    "volume_up":        lambda: pyautogui.press('volumeup',   presses=5, interval=0.02),
    "volume_down":      lambda: pyautogui.press('volumedown', presses=5, interval=0.02),
}

# Minimum time (seconds) between two executions of same gesture to prevent repetition
COOLDOWN = 1.0
_last_execution = {}   # {class: last execution timestamp}


def execute_action(gesture_class):
    """Executes the multimedia action corresponding to the detected gesture."""
    now = time.time()
    if now - _last_execution.get(gesture_class, 0) < COOLDOWN:
        return False   # still in cooldown, don't repeat
    action = ACTIONS.get(gesture_class)
    if action:
        action()
        _last_execution[gesture_class] = now
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# Dataset: black background, white hand (already binarized from source).
# Pipeline: BGR -> Gray -> Resize 64x64 -> Blur -> Otsu -> Normalize -> HOG
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_bgr):
    """
    For dataset images (black background, white hand).
    Session 2: gray -> resize -> blur -> normalization.
    Session 3: HOG features (intelligent flattening).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    res  = cv2.resize(gray, IMG_SIZE)
    blur = cv2.GaussianBlur(res, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    norm = binary / 255.0
    features = hog(norm, **HOG_PARAMS)
    return features


def preprocess_webcam(roi_bgr):
    """
    For webcam frame. Same steps as preprocess_image.
    If the hand occupies more than 50% of the image (light background), invert
    to match dataset format (white hand, black background).
    Returns (feature_vector, binarized_image_for_visualization).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    res  = cv2.resize(gray, IMG_SIZE)
    blur = cv2.GaussianBlur(res, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The dataset has hand=white. If here the hand is black (light background), invert.
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    norm     = binary / 255.0
    features = hog(norm, **HOG_PARAMS)
    return features, binary


# ── Dataset loading ────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _find_image_folders(path):
    """
    Recursively searches for the first layer of subfolders containing
    image files. Works with any structure:
      dataset/clase1/img.jpg        <- single level
      dataset/images/clase1/img.jpg <- two levels (typical kaggle)
    """
    subdirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not subdirs:
        return None, []

    # Check if direct children contain images
    for subdir in subdirs:
        files = os.listdir(os.path.join(path, subdir))
        if any(os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS for f in files):
            # This level already has images -> use it directly
            return path, subdirs

    # If not, go down one level (e.g. dataset/images/ -> dataset/images/class1/)
    if len(subdirs) == 1:
        return _find_image_folders(os.path.join(path, subdirs[0]))

    # Multiple subdirectories without images: try the first one that has children with images
    for subdir in subdirs:
        result, cats = _find_image_folders(os.path.join(path, subdir))
        if cats:
            return result, cats

    return path, subdirs


def load_dataset(base_path):
    X, y = [], []

    class_path, categories = _find_image_folders(base_path)
    if not categories:
        print(f"[ERROR] No class folders found at '{base_path}'")
        return np.array(X), np.array(y)

    categories = sorted(categories)
    print(f"\n[INFO] Loading dataset from '{class_path}'...")

    for category in categories:
        cat_path = os.path.join(class_path, category)
        loaded = 0
        for file_name in os.listdir(cat_path):
            if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(cat_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            X.append(preprocess_image(img))
            y.append(category)
            loaded += 1
        print(f"  OK  {category:22s} -> {loaded} images")

    return np.array(X), np.array(y)


# ── Training ─────────────────────────────────────────────────────────────────
def train_model(X, y):
    print(f"\n[INFO] Features per image: {X.shape[1]}")
    print("[INFO] Splitting dataset 80/20 with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training RandomForestClassifier (200 trees)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluation (Session 3)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred) * 100
    fig, ax = plt.subplots(figsize=(10,8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=clf.classes_,
        cmap='Blues',
        xticks_rotation=45,
        ax=ax
    )
    
    plt.title(f"Confusion Matrix {acc:.2f}%")
    plt.xlabel("Gesture Predicted by AI", fontsize=12)
    plt.ylabel("Real Gesture", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix_final.png")
    plt.show()
    
    print(f"\n{'='*52}")
    print(f"  TEST ACCURACY: {acc * 100:.2f}%")
    print(f"{'='*52}")
    print("\n[DETAILED REPORT]")
    print(classification_report(y_test, y_pred))

    # Confusion matrix in text (Session 3)
    classes = clf.classes_
    cm     = confusion_matrix(y_test, y_pred, labels=classes)
    print("[CONFUSION MATRIX]")
    print(f"{'':24s}" + "  ".join(f"{c[:8]:8s}" for c in classes))
    for i, row in enumerate(cm):
        print(f"  {classes[i]:22s}  " + "  ".join(f"{v:8d}" for v in row))

    joblib.dump(clf, MODEL_PATH)
    print(f"\n[INFO] Model saved to '{MODEL_PATH}'")
    return clf


# ── Real-time detection (Session 4) ──────────────────────────────────────────
def detect_webcam(clf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[INFO] Webcam started. Press Q to exit.")
    print("       Place your hand inside the green rectangle.\n")

    last_gesture = None
    same_counter = 0
    THRESHOLD_FRAMES = 6
    MIN_CONFIDENCE = 0.35
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        x1, y1 = 100, 80
        x2, y2 = 420, 400
        roi_bgr = frame[y1:y2, x1:x2]

        # Prediction
        vector, binary_vis = preprocess_webcam(roi_bgr)
        proba      = clf.predict_proba([vector])[0]
        idx_max    = np.argmax(proba)
        confidence = proba[idx_max]
        pred_class = clf.classes_[idx_max]

        # Temporal smoothing
        if pred_class == last_gesture:
            same_counter += 1
        else:
            last_gesture   = pred_class
            same_counter = 1

        confirmed = (same_counter >= THRESHOLD_FRAMES and confidence >= MIN_CONFIDENCE)

        # HUD - Session 4: visual feedback
        label = GESTURE_LABELS.get(pred_class, pred_class)

        if confirmed:
            color_rect = (0, 255, 0)    # green = OK
        elif confidence >= MIN_CONFIDENCE:
            color_rect = (0, 200, 255)  # yellow = detecting
        else:
            color_rect = (80, 80, 80)   # gray = no confidence

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)

        # Confidence bar below ROI
        bar_width = int((x2 - x1) * confidence)
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 18), color_rect, -1)
        cv2.rectangle(frame, (x1, y2 + 5), (x2,           y2 + 18), (60, 60, 60), 1)

        # Main text
        label_text = label if confirmed else "Waiting for gesture..."
        color_txt = (0, 255, 0) if confirmed else (100, 100, 100)

        cv2.putText(frame, label_text,
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_txt, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.0f}%",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Probability bars by class
        for i, (cls, prob) in enumerate(zip(clf.classes_, proba)):
            etq = GESTURE_LABELS.get(cls, cls)
            bar = int(130 * prob)
            yy  = 100 + i * 22
            cv2.rectangle(frame, (10, yy - 12), (10 + bar, yy), (50, 50, 180), -1)
            cv2.putText(frame, f"{etq}: {prob*100:.0f}%",
                        (12, yy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # AI view in upper right corner
        binary_color  = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
        binary_large = cv2.resize(binary_color, (140, 140))
        frame[5:145, w - 145:w - 5] = binary_large
        cv2.putText(frame, "AI View",
                    (w - 145, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(frame, "Q: Exit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow("Spotify Gesture Detector", frame)

        # Execute action and print when gesture is confirmed (Session 3 + Session 4)
        if confirmed and same_counter == THRESHOLD_FRAMES:
            real_label = GESTURE_LABELS.get(pred_class, pred_class)
            executed = execute_action(pred_class)
            status = "[ACTION SENT]" if executed else "[cooldown]"
            print(f"[GESTURE] {real_label:<22s}  confidence: {confidence*100:.1f}%  {status}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if os.path.exists(MODEL_PATH):
        response = input(
            f"[INFO] Saved model found ('{MODEL_PATH}').\n"
            "       Retrain? (s/N): "
        ).strip().lower()
        retrain = (response == 's')
    else:
        retrain = True

    if retrain:
        X, y = load_dataset(DATASET_PATH)
        print(f"\n[INFO] Total: {X.shape[0]} images | {len(np.unique(y))} classes | {X.shape[1]} features")
        clf = train_model(X, y)
    else:
        clf = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded from '{MODEL_PATH}'")

    detect_webcam(clf)