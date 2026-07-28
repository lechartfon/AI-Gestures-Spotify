import cv2
import os

# Carpeta de origen y destino
CARPETA_ORIGEN = 'dataset-kaggle2/images/siguienteCancion'
CARPETA_DESTINO = 'dataset-kaggle2/images/anteriorCancion'

os.makedirs(CARPETA_DESTINO, exist_ok=True)

imagenes = [
    f for f in os.listdir(CARPETA_ORIGEN)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

for nombre in imagenes:
    ruta = os.path.join(CARPETA_ORIGEN, nombre)
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERROR] No se pudo leer: {ruta}")
        continue

    # 1 = voltear horizontalmente (espejo izquierda-derecha)
    img_volteada = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(CARPETA_DESTINO, nombre), img_volteada)

print(f"Listo. Se voltearon {len(imagenes)} imágenes -> {CARPETA_DESTINO}")
