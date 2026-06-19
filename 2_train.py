import sys
import torch
from pathlib import Path

import ultralytics.utils.checks as _chk
_chk.check_amp = lambda model: False

from ultralytics import YOLO

__author__ = "FoxmoonDev"

BASE_DIR   = Path(__file__).parent.resolve()
YAML_PATH  = BASE_DIR / "car_detection_dataset" / "dataset.yaml"
RUNS_DIR   = BASE_DIR / "runs"
MODEL_SIZE = "yolov8s"
EPOCHS     = 50
IMG_SIZE   = 640
BATCH      = 16
RUN_NAME   = "car_detector"


def check_env():
    print("🔧 Ambiente")
    print(f"  PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU     : {name} ({vram:.1f} GB)")
        return 0
    print("  GPU     : não disponível — usando CPU")
    return "cpu"


def train(device):
    if not YAML_PATH.exists():
        sys.exit(f"❌ {YAML_PATH} não encontrado. Execute 1_prepare_dataset.py primeiro.")

    print(f"\n🚗 FoxmoonDev — Treinamento {MODEL_SIZE}")
    print(f"  Épocas   : {EPOCHS}")
    print(f"  Batch    : {BATCH}")
    print(f"  Img size : {IMG_SIZE}x{IMG_SIZE}\n")

    model = YOLO(f"{MODEL_SIZE}.pt")

    model.train(
        data         = str(YAML_PATH),
        epochs       = EPOCHS,
        imgsz        = IMG_SIZE,
        batch        = BATCH,
        device       = device,
        amp          = False,
        project      = str(RUNS_DIR),
        name         = RUN_NAME,
        exist_ok     = True,
        optimizer    = "AdamW",
        lr0          = 0.001,
        weight_decay = 0.0005,
        patience     = 15,
        augment      = True,
        fliplr       = 0.5,
        mosaic       = 1.0,
        degrees      = 5.0,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        save         = True,
        save_period  = 10,
        plots        = True,
        verbose      = True,
    )


def validate(device):
    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"⚠️  best.pt não encontrado em {best}")
        return
    model   = YOLO(str(best))
    metrics = model.val(data=str(YAML_PATH), imgsz=IMG_SIZE, device=device)
    print("\n📊 MÉTRICAS")
    print("=" * 40)
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precisão : {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")
    print(f"\n  Modelo: {best}")


if __name__ == "__main__":
    device = check_env()
    train(device)
    validate(device)
    print("\n✅ Pronto — execute: python 3_inference.py")
