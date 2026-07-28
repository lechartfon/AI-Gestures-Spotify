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

# pyautogui: desactivar el failsafe para que no interrumpa al mover el raton a la esquina
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # sin pausa entre llamadas para maxima velocidad

# ── Configuracion ──────────────────────────────────────────────────────────────
DATASET_PATH = "dataset-kaggle2/"
MODEL_PATH   = "modelo_gestos.pkl"
IMG_SIZE     = (64, 64)

# HOG params (mismos tanto para dataset como para webcam)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
)

ETIQUETAS = {
    "play_stop":        "Play / Stop",
    "subirVolumen":     "Subir Volumen",
    "bajarVolumen":     "Bajar Volumen",
    "siguienteCancion": "Siguiente Cancion",
    "anteriorCancion":  "Anterior Cancion",
    "up":               "Arriba",
}

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# Dataset: fondo negro, mano blanca (ya binarizado de origen).
# Pipeline: BGR -> Gris -> Resize 64x64 -> Blur -> Otsu -> Normalizar -> HOG
# ──────────────────────────────────────────────────────────────────────────────
def preprocesar_imagen(img_bgr):
    """
    Para imagenes del dataset (fondo negro, mano blanca).
    Sesion 2: grises -> resize -> blur -> normalizacion.
    Sesion 3: HOG features (flattening inteligente).
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    res  = cv2.resize(gris, IMG_SIZE)
    blur = cv2.GaussianBlur(res, (3, 3), 0)
    _, bina = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    norm = bina / 255.0
    features = hog(norm, **HOG_PARAMS)
    return features


def preprocesar_webcam(roi_bgr):
    """
    Para el frame de la webcam. Mismos pasos que preprocesar_imagen.
    Si la mano ocupa mas del 50% de la imagen (fondo claro), se invierte
    para que quede igual que el dataset (mano blanca, fondo negro).
    Devuelve (vector_features, imagen_binarizada_para_visualizar).
    """
    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    res  = cv2.resize(gris, IMG_SIZE)
    blur = cv2.GaussianBlur(res, (5, 5), 0)
    _, bina = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # El dataset tiene mano=blanca. Si aqui la mano es negra (fondo claro), invertir.
    if np.mean(bina) > 127:
        bina = cv2.bitwise_not(bina)

    norm     = bina / 255.0
    features = hog(norm, **HOG_PARAMS)
    return features, bina


# ── Carga del dataset ──────────────────────────────────────────────────────────
EXTENSIONES_IMAGEN = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _encontrar_carpetas_con_imagenes(ruta):
    """
    Busca recursivamente la primera capa de subcarpetas que contenga
    archivos de imagen. Asi funciona con cualquier estructura:
      dataset/clase1/img.jpg        <- un nivel
      dataset/images/clase1/img.jpg <- dos niveles (kaggle tipico)
    """
    hijos = [
        d for d in os.listdir(ruta)
        if os.path.isdir(os.path.join(ruta, d))
    ]
    if not hijos:
        return None, []

    # Comprobar si los hijos directos contienen imagenes
    for hijo in hijos:
        archivos = os.listdir(os.path.join(ruta, hijo))
        if any(os.path.splitext(f)[1].lower() in EXTENSIONES_IMAGEN for f in archivos):
            # Este nivel YA tiene imagenes -> usarlo directamente
            return ruta, hijos

    # Si no, bajar un nivel (ej: dataset/images/ -> dataset/images/clase1/)
    if len(hijos) == 1:
        return _encontrar_carpetas_con_imagenes(os.path.join(ruta, hijos[0]))

    # Varios subdirectorios sin imagenes: intentar el primero que tenga hijos con imagenes
    for hijo in hijos:
        resultado, cats = _encontrar_carpetas_con_imagenes(os.path.join(ruta, hijo))
        if cats:
            return resultado, cats

    return ruta, hijos


def cargar_dataset(ruta_base):
    X, y = [], []

    ruta_clases, categorias = _encontrar_carpetas_con_imagenes(ruta_base)
    if not categorias:
        print(f"[ERROR] No se encontraron carpetas de clases en '{ruta_base}'")
        return np.array(X), np.array(y)

    categorias = sorted(categorias)
    print(f"\n[INFO] Cargando dataset desde '{ruta_clases}'...")

    for categoria in categorias:
        ruta_cat = os.path.join(ruta_clases, categoria)
        cargadas = 0
        for archivo in os.listdir(ruta_cat):
            if os.path.splitext(archivo)[1].lower() not in EXTENSIONES_IMAGEN:
                continue
            ruta_img = os.path.join(ruta_cat, archivo)
            img = cv2.imread(ruta_img)
            if img is None:
                continue
            X.append(preprocesar_imagen(img))
            y.append(categoria)
            cargadas += 1
        print(f"  OK  {categoria:22s} -> {cargadas} imagenes")

    return np.array(X), np.array(y)


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar_modelo(X, y):
    print(f"\n[INFO] Features por imagen: {X.shape[1]}")
    print("[INFO] Dividiendo dataset 80/20 con estratificacion...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Entrenando RandomForestClassifier (200 arboles)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluacion (Sesion 3)
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
    
    plt.title(f"Matriz de Confusión {acc:.2f}%")
    plt.xlabel("Gesto Predicho por la IA", fontsize=12)
    plt.ylabel("Gesto Real", fontsize=12)
    plt.tight_layout()
    plt.savefig("Matriz_confusion_final.png")
    plt.show()
    
    print(f"\n{'='*52}")
    print(f"  EXACTITUD EN TEST: {acc * 100:.2f}%")
    print(f"{'='*52}")
    print("\n[REPORTE DETALLADO]")
    print(classification_report(y_test, y_pred))

    # Matriz de confusion en texto (Sesion 3)
    clases = clf.classes_
    cm     = confusion_matrix(y_test, y_pred, labels=clases)
    print("[MATRIZ DE CONFUSION]")
    print(f"{'':24s}" + "  ".join(f"{c[:8]:8s}" for c in clases))
    for i, fila in enumerate(cm):
        print(f"  {clases[i]:22s}  " + "  ".join(f"{v:8d}" for v in fila))

    joblib.dump(clf, MODEL_PATH)
    print(f"\n[INFO] Modelo guardado en '{MODEL_PATH}'")
    return clf


# ── Acciones de Spotify via pyautogui ────────────────────────────────────────
# Teclas multimedia de Spotify (funcionan con Spotify en segundo plano en Windows)
ACCIONES = {
    "play_stop":        lambda: pyautogui.press('playpause'),
    "siguienteCancion": lambda: pyautogui.press('nexttrack'),
    "anteriorCancion":  lambda: pyautogui.press('prevtrack'),
    "subirVolumen":     lambda: pyautogui.press('volumeup',   presses=5, interval=0.02),
    "bajarVolumen":     lambda: pyautogui.press('volumedown', presses=5, interval=0.02),
    "up":               lambda: pyautogui.press('playpause'),
}

# Tiempo minimo (segundos) entre dos ejecuciones del mismo gesto para no repetir
COOLDOWN = 1.0
_ultimo_tiempo = {}   # {clase: timestamp ultima ejecucion}


def ejecutar_accion(clase):
    """Ejecuta la accion multimedia correspondiente al gesto detectado."""
    ahora = time.time()
    if ahora - _ultimo_tiempo.get(clase, 0) < COOLDOWN:
        return False   # aun en cooldown, no repetir
    accion = ACCIONES.get(clase)
    if accion:
        accion()
        _ultimo_tiempo[clase] = ahora
        return True
    return False


# ── Deteccion en tiempo real (Sesion 4) ───────────────────────────────────────
def detectar_webcam(clf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se puede abrir la webcam.")
        return

    print("\n[INFO] Webcam iniciada. Presiona Q para salir.")
    print("       Coloca la mano dentro del rectangulo verde.\n")

    ultimo_gesto   = None
    contador_igual = 0
    UMBRAL_FRAMES  = 6
    CONFIANZA_MIN  = 0.35
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        x1, y1 = 100, 80
        x2, y2 = 420, 400
        roi_bgr = frame[y1:y2, x1:x2]

        # Prediccion
        vector, bina_vis = preprocesar_webcam(roi_bgr)
        proba      = clf.predict_proba([vector])[0]
        idx_max    = np.argmax(proba)
        confianza  = proba[idx_max]
        clase_pred = clf.classes_[idx_max]

        # Suavizado temporal
        if clase_pred == ultimo_gesto:
            contador_igual += 1
        else:
            ultimo_gesto   = clase_pred
            contador_igual = 1

        confirmado = (contador_igual >= UMBRAL_FRAMES and confianza >= CONFIANZA_MIN)

        # HUD - Sesion 4: feedback visual
        etiqueta = ETIQUETAS.get(clase_pred, clase_pred)

        if confirmado:
            color_rect = (0, 255, 0)    # verde = OK
        elif confianza >= CONFIANZA_MIN:
            color_rect = (0, 200, 255)  # amarillo = detectando
        else:
            color_rect = (80, 80, 80)   # gris = sin confianza

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)

        # Barra de confianza debajo del ROI
        barra_w = int((x2 - x1) * confianza)
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + barra_w, y2 + 18), color_rect, -1)
        cv2.rectangle(frame, (x1, y2 + 5), (x2,           y2 + 18), (60, 60, 60), 1)

        # Texto principal
        etiqueta_display = etiqueta if confirmado else "Esperando gesto..."
        color_txt = (0, 255, 0) if confirmado else (100, 100, 100)

        cv2.putText(frame, etiqueta_display,
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_txt, 2)
        cv2.putText(frame, f"Confianza: {confianza*100:.0f}%",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Barras de probabilidad por clase
        for i, (cls, prob) in enumerate(zip(clf.classes_, proba)):
            etq = ETIQUETAS.get(cls, cls)
            bar = int(130 * prob)
            yy  = 100 + i * 22
            cv2.rectangle(frame, (10, yy - 12), (10 + bar, yy), (50, 50, 180), -1)
            cv2.putText(frame, f"{etq}: {prob*100:.0f}%",
                        (12, yy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Vista IA en esquina superior derecha
        bina_color  = cv2.cvtColor(bina_vis, cv2.COLOR_GRAY2BGR)
        bina_grande = cv2.resize(bina_color, (140, 140))
        frame[5:145, w - 145:w - 5] = bina_grande
        cv2.putText(frame, "Vista IA",
                    (w - 145, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(frame, "Q: Salir",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow("Detector de Gestos Spotify", frame)

        # Ejecutar accion y print al confirmar gesto (Sesion 3 + Sesion 4)
        if confirmado and contador_igual == UMBRAL_FRAMES:
            etiqueta_real = ETIQUETAS.get(clase_pred, clase_pred)
            ejecutado = ejecutar_accion(clase_pred)
            estado = "[ACCION ENVIADA]" if ejecutado else "[cooldown]"
            print(f"[GESTO] {etiqueta_real:<22s}  confianza: {confianza*100:.1f}%  {estado}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if os.path.exists(MODEL_PATH):
        resp = input(
            f"[INFO] Modelo guardado encontrado ('{MODEL_PATH}').\n"
            "       Re-entrenar? (s/N): "
        ).strip().lower()
        reentrenar = (resp == 's')
    else:
        reentrenar = True

    if reentrenar:
        X, y = cargar_dataset(DATASET_PATH)
        print(f"\n[INFO] Total: {X.shape[0]} imagenes | {len(np.unique(y))} clases | {X.shape[1]} features")
        clf = entrenar_modelo(X, y)
    else:
        clf = joblib.load(MODEL_PATH)
        print(f"[INFO] Modelo cargado desde '{MODEL_PATH}'")

    detectar_webcam(clf)

