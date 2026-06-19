import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

__author__ = "FoxmoonDev"

BASE_DIR   = Path(__file__).parent.resolve()
DATASET2   = BASE_DIR / "data2"
OUTPUT_DIR = BASE_DIR / "car_detection_dataset"
FRAME_STEP = 5


def bbox_to_yolo(w, h, xmin, ymin, xmax, ymax):
    xc = ((xmin + xmax) / 2) / w
    yc = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return round(xc,6), round(yc,6), round(bw,6), round(bh,6)


def poly_to_bbox(pts):
    arr = np.array(pts, dtype=float).reshape(-1, 2)
    return arr[:,0].min(), arr[:,1].min(), arr[:,0].max(), arr[:,1].max()


def extract_frames(src_dir, dst_dir, step):
    videos = (
        list(src_dir.rglob("*.mp4")) +
        list(src_dir.rglob("*.avi")) +
        list(src_dir.rglob("*.mov"))
    )
    if not videos:
        print("  Nenhum vídeo encontrado.")
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for vid in videos:
        cap  = cv2.VideoCapture(str(vid))
        fidx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % step == 0:
                name = f"{vid.stem}_f{fidx:06d}.jpg"
                path = dst_dir / name
                cv2.imwrite(str(path), frame)
                saved.append(path)
            fidx += 1
        cap.release()
        print(f"  {vid.name}: {fidx // step} frames")
    print(f"  Total: {len(saved)} frames extraídos")
    return saved


def process_coco(json_files, search_dir):
    converted = 0
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        id2img  = {img["id"]: img for img in data.get("images", [])}
        anns_by = {}
        for ann in data.get("annotations", []):
            anns_by.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in tqdm(anns_by.items(), desc=f"  {jf.name}"):
            info = id2img.get(img_id)
            if not info:
                continue
            iw, ih = info["width"], info["height"]
            fname  = Path(info["file_name"]).name

            matches = list(search_dir.rglob(fname))
            if matches:
                dst_img = OUTPUT_DIR / "images" / "train" / fname
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(matches[0], dst_img)

            lbl = OUTPUT_DIR / "labels" / "train" / (Path(fname).stem + ".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            with open(lbl, "w") as lf:
                for ann in anns:
                    if "segmentation" in ann and ann["segmentation"]:
                        seg = ann["segmentation"]
                        pts = seg[0] if isinstance(seg, list) else []
                        if not pts:
                            continue
                        xmin, ymin, xmax, ymax = poly_to_bbox(pts)
                    elif "bbox" in ann:
                        bx, by, bw, bh = ann["bbox"]
                        xmin, ymin, xmax, ymax = bx, by, bx+bw, by+bh
                    else:
                        continue
                    xc, yc, nw, nh = bbox_to_yolo(iw, ih, xmin, ymin, xmax, ymax)
                    lf.write(f"0 {xc} {yc} {nw} {nh}\n")
                    converted += 1
    print(f"  {converted} anotações convertidas")


def process_yolo_txt(txt_files):
    copied = 0
    for txt in tqdm(txt_files, desc="  YOLO TXT"):
        dst = OUTPUT_DIR / "labels" / "train" / txt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt, dst)
        for ext in (".jpg", ".png", ".jpeg"):
            img = txt.with_suffix(ext)
            if img.exists():
                dst_img = OUTPUT_DIR / "images" / "train" / img.name
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst_img)
                copied += 1
                break
    print(f"  {copied} pares imagem+label copiados")


def update_yaml():
    n = {s: len(list((OUTPUT_DIR / "images" / s).glob("*")))
         for s in ("train", "val", "test")}
    content = (
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: car\n"
    )
    (OUTPUT_DIR / "dataset.yaml").write_text(content)
    print(f"  dataset.yaml atualizado | treino:{n['train']} val:{n['val']}")


if __name__ == "__main__":
    print("🚗 FoxmoonDev — Preparação Dataset 2")
    print("=" * 40)

    if not DATASET2.exists():
        sys.exit(
            f"❌ Pasta não encontrada: {DATASET2}\n"
            "   Baixe: https://www.kaggle.com/datasets/trainingdatapro/cars-video-object-tracking\n"
            "   e extraia em ./data2/"
        )

    frames_dir = BASE_DIR / "data2_frames"
    print("\n  Extraindo frames...")
    extract_frames(DATASET2, frames_dir, step=FRAME_STEP)

    json_files = list(DATASET2.rglob("*.json"))
    txt_files  = [t for t in DATASET2.rglob("*.txt") if not t.name.startswith(".")]

    print(f"\n  JSON (COCO) : {len(json_files)}")
    print(f"  TXT  (YOLO) : {len(txt_files)}")

    if json_files:
        process_coco(json_files, DATASET2)
    elif txt_files:
        process_yolo_txt(txt_files)
    else:
        print("  Nenhuma anotação encontrada — apenas frames extraídos.")

    update_yaml()
    print("\n✅ Pronto — execute: python 2_train.py")
