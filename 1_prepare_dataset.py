import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

__author__ = "FoxmoonDev"

BASE_DIR     = Path(__file__).parent.resolve()
DATASET_DIR  = BASE_DIR / "data"
OUTPUT_DIR   = BASE_DIR / "car_detection_dataset"
VAL_SPLIT    = 0.15
SEED         = 42
TRAIN_IMAGES = DATASET_DIR / "training_images"
TEST_IMAGES  = DATASET_DIR / "testing_images"
CSV_CANDIDATES = [
    DATASET_DIR / "train_solution_bounding_boxes (1).csv",
    DATASET_DIR / "train_solution_bounding_boxes.csv",
]


def find_csv():
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    found = list(DATASET_DIR.glob("*.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"CSV não encontrado em {DATASET_DIR}\n"
        "Baixe: https://www.kaggle.com/datasets/sshikamaru/car-object-detection"
    )


def bbox_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax):
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)


def explore(df):
    print("\n📊 DATASET")
    print("=" * 40)
    print(f"  Anotações  : {len(df)}")
    print(f"  Imagens    : {df['image'].nunique()}")
    print(f"  Carros/img : {len(df) / df['image'].nunique():.1f}")
    df["w"] = df["xmax"] - df["xmin"]
    df["h"] = df["ymax"] - df["ymin"]
    print(f"  Bbox média : {df['w'].mean():.0f}x{df['h'].mean():.0f} px")


def visualize(df, images_dir, n=4):
    imgs = df["image"].unique()[:n]
    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, name in zip(axes, imgs):
        path = images_dir / name
        img = cv2.imread(str(path))
        if img is None:
            ax.set_title(f"Não encontrado:\n{name}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        for _, r in df[df["image"] == name].iterrows():
            rect = patches.Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
        ax.set_title(f"{name}\n({len(df[df['image'] == name])} carros)", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out = BASE_DIR / "sample_annotations.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Amostras salvas: {out}")


def create_structure():
    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_split(df, images_dir, split):
    for img_name in tqdm(df["image"].unique(), desc=f"  {split:5s}"):
        src = images_dir / img_name
        if not src.exists():
            continue
        shutil.copy2(src, OUTPUT_DIR / "images" / split / img_name)
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_file = OUTPUT_DIR / "labels" / split / (Path(img_name).stem + ".txt")
        with open(label_file, "w") as f:
            for _, r in df[df["image"] == img_name].iterrows():
                xc, yc, bw, bh = bbox_to_yolo(w, h, r["xmin"], r["ymin"], r["xmax"], r["ymax"])
                f.write(f"0 {xc} {yc} {bw} {bh}\n")


def process_test(test_dir):
    imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    for src in tqdm(imgs, desc="  test "):
        shutil.copy2(src, OUTPUT_DIR / "images" / "test" / src.name)
    print(f"  {len(imgs)} imagens de teste copiadas")


def write_yaml():
    n_train = len(list((OUTPUT_DIR / "images" / "train").iterdir()))
    n_val   = len(list((OUTPUT_DIR / "images" / "val").iterdir()))
    n_test  = len(list((OUTPUT_DIR / "images" / "test").iterdir()))
    content = (
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: car\n"
    )
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(content)
    print(f"  dataset.yaml | treino:{n_train} val:{n_val} test:{n_test}")
    return yaml_path


if __name__ == "__main__":
    print("🚗 FoxmoonDev — Preparação do Dataset")
    print("=" * 40)

    if not TRAIN_IMAGES.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {TRAIN_IMAGES}\n"
            "Extraia o Dataset 1 em ./data/"
        )

    csv_path = find_csv()
    print(f"  CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)
    if "fname" in df.columns:
        df = df.rename(columns={"fname": "image"})

    explore(df)
    visualize(df, TRAIN_IMAGES)
    create_structure()

    all_imgs = df["image"].unique()
    train_imgs, val_imgs = train_test_split(all_imgs, test_size=VAL_SPLIT, random_state=SEED)
    print(f"\n  Split: {len(train_imgs)} treino | {len(val_imgs)} validação")

    process_split(df[df["image"].isin(train_imgs)], TRAIN_IMAGES, "train")
    process_split(df[df["image"].isin(val_imgs)],   TRAIN_IMAGES, "val")

    if TEST_IMAGES.exists():
        process_test(TEST_IMAGES)

    write_yaml()
    print("\n✅ Pronto — execute: python 2_train.py")
