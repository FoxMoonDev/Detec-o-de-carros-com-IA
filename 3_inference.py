import argparse
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import ultralytics.utils.checks as _chk
_chk.check_amp = lambda model: False

from ultralytics import YOLO

__author__ = "FoxmoonDev"

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs"
WEIGHTS  = RUNS_DIR / "car_detector" / "weights" / "best.pt"
TEST_DIR = BASE_DIR / "car_detection_dataset" / "images" / "test"
OUT_DIR  = BASE_DIR / "predictions"
CONF     = 0.25
IOU      = 0.45
IMG_SIZE = 640


def find_weights():
    if WEIGHTS.exists():
        return WEIGHTS
    candidates = sorted(
        RUNS_DIR.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if candidates:
        print(f"  Modelo encontrado: {candidates[0]}")
        return candidates[0]
    sys.exit(
        "❌ Nenhum best.pt encontrado.\n"
        "   Execute 2_train.py primeiro."
    )


def predict_folder(model, source, n=16):
    source = Path(source)
    imgs   = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))[:n]
    if not imgs:
        sys.exit(f"❌ Nenhuma imagem encontrada em {source}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols  = 4
    rows  = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes  = np.array(axes).flatten()
    total = 0

    for idx, img_path in enumerate(imgs):
        res   = model.predict(source=str(img_path), conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]
        n_det = len(res.boxes)
        total += n_det

        annotated = cv2.cvtColor(res.plot(line_width=2), cv2.COLOR_BGR2RGB)
        axes[idx].imshow(annotated)
        axes[idx].set_title(f"{img_path.name}\n{n_det} carro(s)", fontsize=8)
        axes[idx].axis("off")

        cv2.imwrite(str(OUT_DIR / img_path.name), res.plot(line_width=2))

    for i in range(len(imgs), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"FoxmoonDev — Car Detection | {total} carros em {len(imgs)} imagens", fontsize=12)
    plt.tight_layout()
    grid_path = OUT_DIR / "detection_grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  Grid salvo     : {grid_path}")
    print(f"  Imagens salvas : {OUT_DIR}/")
    print(f"  Total detectado: {total} carros")


def predict_single(model, img_path):
    img_path = Path(img_path)
    if not img_path.exists():
        sys.exit(f"❌ Arquivo não encontrado: {img_path}")

    res = model.predict(source=str(img_path), conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]

    print(f"\n  Imagem: {img_path.name}")
    print(f"  Carros: {len(res.boxes)}")
    for i, box in enumerate(res.boxes):
        conf = box.conf[0].item()
        xyxy = [round(v) for v in box.xyxy[0].tolist()]
        print(f"  [{i+1}] conf={conf:.2f}  bbox={xyxy}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), res.plot(line_width=2))
    print(f"  Salvo: {out_path}")


def export_model(model):
    print("  Exportando para ONNX...")
    model.export(format="onnx", imgsz=IMG_SIZE, dynamic=True)
    print("  Export concluído")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoxmoonDev — Car Detection Inference")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    print("🚗 FoxmoonDev — Inferência")
    print("=" * 40)

    weights = find_weights()
    model   = YOLO(str(weights))
    print(f"  Modelo: {weights}")

    if args.export:
        export_model(model)
    elif args.source:
        src = Path(args.source)
        if src.is_dir():
            predict_folder(model, src, n=args.n)
        else:
            predict_single(model, src)
    else:
        if not TEST_DIR.exists():
            sys.exit(f"❌ Pasta de teste não encontrada: {TEST_DIR}")
        predict_folder(model, TEST_DIR, n=args.n)

    print("\n✅ Concluído")
