"""
utils/dataset_tools.py
────────────────────────
Utilities for preprocessing and augmentation of image datasets.

Usage:
        python utils/dataset_tools.py resize <source> [destination] [width] [height]
        python utils/dataset_tools.py augment <source> <destination> [--rotate 90|180|270] [--flip]

    Examples:
        python utils/dataset_tools.py resize dataset-kaggle2
        python utils/dataset_tools.py resize dataset-kaggle2 dataset-kaggle2-64 64 64
        python utils/dataset_tools.py augment dataset-kaggle2/images/siguienteCancion dataset-kaggle2/images/anteriorCancion --rotate 180 --flip
"""

import cv2
import os
import sys
import argparse

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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
        if any(os.path.splitext(f)[1].lower() in EXTENSIONS for f in files):
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


def resize_dataset(source_dir, destination, width=64, height=64):
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source folder does not exist: '{source_dir}'")
        return

    image_folders, classes = _find_image_folders(source_dir)
    if not classes:
        print(f"[ERROR] No class folders found at '{source_dir}'")
        return

    print(f"[INFO] Source:      {image_folders}")
    print(f"[INFO] Destination: {destination}")
    print(f"[INFO] Size:        {width}x{height}")
    print(f"[INFO] Classes:     {sorted(classes)}\n")

    total = 0
    for class_name in sorted(classes):
        src = os.path.join(image_folders, class_name)
        dst = os.path.join(destination, class_name)
        os.makedirs(dst, exist_ok=True)

        files = [
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ]
        processed = 0
        for file_name in files:
            img = cv2.imread(os.path.join(src, file_name))
            if img is None:
                print(f"  [WARN] Could not read: {file_name}")
                continue
            img_res = cv2.resize(img, (width, height))
            cv2.imwrite(os.path.join(dst, file_name), img_res)
            processed += 1

        print(f"  OK  {class_name:25s} -> {processed:4d} images")
        total += processed

    print(f"\n[INFO] Total processed: {total} images saved to '{destination}'")


def augment_dataset(source, destination, rotate=None, flip=False):
    """
    Increases the dataset by applying rotations and/or horizontal flip.
    The augmented images are saved in 'destination' with suffix.
    """
    if not os.path.isdir(source):
        print(f"[ERROR] Source folder does not exist: '{source}'")
        return

    os.makedirs(destination, exist_ok=True)

    files = [
        f for f in os.listdir(source)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ]

    if not files:
        print(f"[WARN] No images in '{source}'")
        return

    print(f"[INFO] Augmenting {len(files)} images from '{source}' -> '{destination}'")

    for file_name in files:
        file_path = os.path.join(source, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Could not read: {file_name}")
            continue

        file_base, file_ext = os.path.splitext(file_name)

        # Original
        cv2.imwrite(os.path.join(destination, file_name), img)

        # Rotations
        if rotate in (90, 180, 270):
            rot_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            img_rot = cv2.rotate(img, rot_map[rotate])
            cv2.imwrite(os.path.join(destination, f"{file_base}_rot{rotate}{file_ext}"), img_rot)

        # Horizontal flip (mirror)
        if flip:
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(destination, f"{file_base}_flip{file_ext}"), img_flip)

    print(f"[INFO] Augmentation completed. Images in '{destination}'")


def main():
    parser = argparse.ArgumentParser(
        description="Utilities for image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: resize
    p_resize = subparsers.add_parser("resize", help="Resize images to fixed size")
    p_resize.add_argument("source", help="Source folder (supports nested structure)")
    p_resize.add_argument("destination", nargs="?", help="Destination folder (default: source_64)")
    p_resize.add_argument("width", nargs="?", type=int, default=64, help="Width (default: 64)")
    p_resize.add_argument("height", nargs="?", type=int, default=64, help="Height (default: 64)")

    # Subcommand: augment
    p_aug = subparsers.add_parser("augment", help="Data augmentation (rotation, flip)")
    p_aug.add_argument("source", help="Source folder with images")
    p_aug.add_argument("destination", help="Destination folder for augmented images")
    p_aug.add_argument("--rotate", type=int, choices=[90, 180, 270], help="Rotation in degrees")
    p_aug.add_argument("--flip", action="store_true", help="Horizontal flip (mirror)")

    args = parser.parse_args()

    if args.command == "resize":
        destination = args.destination or args.source.rstrip('/\\') + "_64"
        resize_dataset(args.source, destination, args.width, args.height)
    elif args.command == "augment":
        if not args.rotate and not args.flip:
            print("[ERROR] Specify at least --rotate or --flip")
            sys.exit(1)
        augment_dataset(args.source, args.destination, args.rotate, args.flip)


if __name__ == "__main__":
    main()