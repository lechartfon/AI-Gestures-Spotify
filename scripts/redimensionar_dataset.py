"""
redimensionar_dataset.py
────────────────────────
Uso:
    python redimensionar_dataset.py <carpeta_origen> [carpeta_destino] [ancho] [alto]

Ejemplos:
    python redimensionar_dataset.py dataset-kaggle2
    python redimensionar_dataset.py dataset-kaggle2 dataset-kaggle2-64 64 64

Si no se indica destino, crea una carpeta con sufijo '_64' junto al origen.
Soporta cualquier estructura de carpetas (con o sin subcarpeta 'images/').
Redimensiona a 64x64 por defecto.
"""

import cv2
import os
import sys

EXTENSIONES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _encontrar_carpetas_con_imagenes(ruta):
    """Igual que en detector_gestos: desciende hasta encontrar clases con imagenes."""
    hijos = [
        d for d in os.listdir(ruta)
        if os.path.isdir(os.path.join(ruta, d))
    ]
    if not hijos:
        return None, []

    for hijo in hijos:
        archivos = os.listdir(os.path.join(ruta, hijo))
        if any(os.path.splitext(f)[1].lower() in EXTENSIONES for f in archivos):
            return ruta, hijos

    if len(hijos) == 1:
        return _encontrar_carpetas_con_imagenes(os.path.join(ruta, hijos[0]))

    for hijo in hijos:
        resultado, cats = _encontrar_carpetas_con_imagenes(os.path.join(ruta, hijo))
        if cats:
            return resultado, cats

    return ruta, hijos


def redimensionar_dataset(origen, destino, ancho=64, alto=64):
    if not os.path.isdir(origen):
        print(f"[ERROR] La carpeta de origen no existe: '{origen}'")
        return

    ruta_clases, categorias = _encontrar_carpetas_con_imagenes(origen)
    if not categorias:
        print(f"[ERROR] No se encontraron carpetas de clases en '{origen}'")
        return

    print(f"[INFO] Origen:  {ruta_clases}")
    print(f"[INFO] Destino: {destino}")
    print(f"[INFO] Tamaño:  {ancho}x{alto}")
    print(f"[INFO] Clases:  {sorted(categorias)}\n")

    total = 0
    for categoria in sorted(categorias):
        src = os.path.join(ruta_clases, categoria)
        dst = os.path.join(destino, categoria)
        os.makedirs(dst, exist_ok=True)

        archivos = [
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in EXTENSIONES
        ]
        procesadas = 0
        for archivo in archivos:
            img = cv2.imread(os.path.join(src, archivo))
            if img is None:
                print(f"  [WARN] No se pudo leer: {archivo}")
                continue
            img_res = cv2.resize(img, (ancho, alto))
            cv2.imwrite(os.path.join(dst, archivo), img_res)
            procesadas += 1

        print(f"  OK  {categoria:25s} -> {procesadas:4d} imagenes")
        total += procesadas

    print(f"\n[INFO] Total procesadas: {total} imagenes guardadas en '{destino}'")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Uso: python redimensionar_dataset.py <origen> [destino] [ancho] [alto]")
        print("Ejemplo: python redimensionar_dataset.py dataset-kaggle2")
        sys.exit(1)

    origen  = args[0].rstrip('/\\')
    destino = args[1] if len(args) > 1 else origen + "_64"
    ancho   = int(args[2]) if len(args) > 2 else 64
    alto    = int(args[3]) if len(args) > 3 else 64

    redimensionar_dataset(origen, destino, ancho, alto)
