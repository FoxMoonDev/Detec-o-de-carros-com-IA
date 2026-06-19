import argparse
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from pathlib import Path

import ultralytics.utils.checks as _chk
_chk.check_amp = lambda model: False

from ultralytics import YOLO

__author__ = "FoxmoonDev"

BASE_DIR  = Path(__file__).parent.resolve()
RUNS_DIR  = BASE_DIR / "runs"
WEIGHTS   = RUNS_DIR / "car_detector" / "weights" / "best.pt"
OUT_DIR   = BASE_DIR / "tracking_output"

CONF             = 0.30
IOU              = 0.45
IMG_SIZE         = 640
TRACKER          = "bytetrack.yaml"
TRAIL_LEN        = 40
COUNT_LINE_FRAC  = 0.5
PIXELS_PER_METER = 15.0
FPS_FALLBACK     = 25

PALETTE = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,200,0),(150,0,255),(0,180,100),(255,80,80),(80,80,255),
]


def get_color(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]


def find_weights():
    if WEIGHTS.exists():
        return WEIGHTS
    candidates = sorted(
        RUNS_DIR.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if candidates:
        print(f"  Modelo: {candidates[0]}")
        return candidates[0]
    sys.exit("❌ best.pt não encontrado. Execute 2_train.py primeiro.")


def open_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        sys.exit(f"❌ Não foi possível abrir: {source}")
    return cap


class SpeedEstimator:
    def __init__(self, ppm, fps):
        self.ppm = ppm
        self.fps = fps
        self.positions = defaultdict(lambda: deque(maxlen=10))

    def update(self, track_id, cx, cy):
        self.positions[track_id].append((cx, cy))

    def kmh(self, track_id):
        pts = self.positions[track_id]
        if len(pts) < 2:
            return 0.0
        dists = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1, len(pts))]
        return (np.mean(dists) * self.fps / self.ppm) * 3.6


class LineCounter:
    def __init__(self, frac, frame_h):
        self.y       = int(frac * frame_h)
        self.prev    = {}
        self.counted = set()
        self.up      = 0
        self.down    = 0

    def update(self, track_id, cy):
        if track_id in self.counted:
            return
        if track_id in self.prev:
            if self.prev[track_id] < self.y <= cy:
                self.down += 1
                self.counted.add(track_id)
            elif self.prev[track_id] > self.y >= cy:
                self.up += 1
                self.counted.add(track_id)
        self.prev[track_id] = cy

    @property
    def total(self):
        return self.up + self.down


class Heatmap:
    def __init__(self, h, w):
        self.map = np.zeros((h, w), np.float32)

    def update(self, cx, cy, radius=12):
        if 0 <= cy < self.map.shape[0] and 0 <= cx < self.map.shape[1]:
            cv2.circle(self.map, (cx, cy), radius, 1.0, -1)

    def overlay(self, frame, alpha=0.45):
        if self.map.max() == 0:
            return frame
        norm = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur = cv2.GaussianBlur(norm, (31, 31), 0)
        cmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        mask = blur > 10
        out  = frame.copy()
        out[mask] = cv2.addWeighted(frame, 1-alpha, cmap, alpha, 0)[mask]
        return out

    def save(self, path):
        norm = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur = cv2.GaussianBlur(norm, (51, 51), 0)
        cv2.imwrite(str(path), cv2.applyColorMap(blur, cv2.COLORMAP_JET))
        print(f"  Heatmap: {path}")


def draw_overlay(frame, frame_idx, total_frames, counter, active):
    lines = [
        f"Frame: {frame_idx}" + (f"/{total_frames}" if total_frames else ""),
        f"Ativos: {active}",
        f"Contados: {counter.total}  (↓{counter.down} ↑{counter.up})",
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)


def save_trajectory_plot(df, w, h, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    for tid, grp in df.groupby("track_id"):
        c = [v/255 for v in get_color(tid)[::-1]]
        ax.plot(grp["cx"], grp["cy"], color=c, lw=1.2, alpha=0.85)
        ax.scatter(grp["cx"].iloc[0], grp["cy"].iloc[0], color=c, s=25, zorder=5)
        ax.annotate(f"#{tid}", (grp["cx"].iloc[-1], grp["cy"].iloc[-1]), color=c, fontsize=7)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title("FoxmoonDev — Trajetórias", color="white", fontsize=12)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Trajetórias: {out_path}")


def run(source, show_heatmap=False, show_window=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(find_weights()))

    cap   = open_source(source)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    src_name  = Path(str(source)).stem if not str(source).startswith("rtsp") else "stream"
    out_video = OUT_DIR / f"{src_name}_tracked.mp4"
    writer    = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (W, H)
    )

    counter   = LineCounter(COUNT_LINE_FRAC, H)
    speed_est = SpeedEstimator(PIXELS_PER_METER, FPS)
    hmap      = Heatmap(H, W)
    trails    = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    records   = []
    fidx      = 0

    print(f"\n  Source: {source}  [{W}x{H} @ {FPS:.0f}fps]")
    print("  Pressione 'q' para parar.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        results   = model.track(
            source=frame, conf=CONF, iou=IOU,
            imgsz=IMG_SIZE, tracker=TRACKER,
            persist=True, verbose=False
        )[0]
        annotated = frame.copy()

        ly = int(COUNT_LINE_FRAC * H)
        cv2.line(annotated, (0, ly), (W, ly), (0, 200, 255), 2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids   = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1+x2)//2, (y1+y2)//2
                c      = get_color(tid)

                trails[tid].append((cx, cy))
                counter.update(tid, cy)
                speed_est.update(tid, cx, cy)
                hmap.update(cx, cy)

                spd = speed_est.kmh(tid)

                cv2.rectangle(annotated, (x1,y1), (x2,y2), c, 2)

                label = f"#{tid}  {spd:.0f}km/h"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1-lh-6), (x1+lw+4, y1), c, -1)
                cv2.putText(annotated, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                pts = list(trails[tid])
                for i in range(1, len(pts)):
                    fade = tuple(int(v * i / len(pts)) for v in c)
                    cv2.line(annotated, pts[i-1], pts[i], fade, 2)
                cv2.circle(annotated, (cx, cy), 4, c, -1)

                records.append({
                    "frame": fidx, "track_id": int(tid),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "cx": cx, "cy": cy,
                    "conf": round(float(conf), 3),
                    "speed_kmh": round(spd, 1),
                })

        if show_heatmap:
            annotated = hmap.overlay(annotated)

        active = len(results.boxes.id) if results.boxes.id is not None else 0
        draw_overlay(annotated, fidx, TOTAL, counter, active)
        writer.write(annotated)

        if show_window:
            cv2.imshow("FoxmoonDev — Car Tracking  [q=sair]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrompido.")
                break

        if fidx % 100 == 0:
            print(f"  Frame {fidx}" + (f"/{TOTAL}" if TOTAL else "") +
                  f" | contados: {counter.total}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n  Vídeo salvo: {out_video}")

    if records:
        df       = pd.DataFrame(records)
        csv_path = OUT_DIR / f"{src_name}_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV salvo  : {csv_path}")
        hmap.save(OUT_DIR / f"{src_name}_heatmap.jpg")
        save_trajectory_plot(df, W, H, OUT_DIR / f"{src_name}_trajectories.png")

        print("\n📊 RELATÓRIO")
        print(f"  IDs únicos : {df['track_id'].nunique()}")
        print(f"  Contados   : {counter.total}  (↓{counter.down} ↑{counter.up})")
        spds = df[df["speed_kmh"] > 1]["speed_kmh"]
        if len(spds):
            print(f"  Vel. média : {spds.mean():.1f} km/h")
            print(f"  Vel. máx   : {spds.max():.1f} km/h")
    else:
        print("  Nenhuma detecção registrada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoxmoonDev — Car Tracking")
    parser.add_argument("--source",    required=True, help="Vídeo, URL RTSP ou índice de webcam (ex: 0)")
    parser.add_argument("--heatmap",   action="store_true", help="Mostrar heatmap em tempo real")
    parser.add_argument("--no-window", action="store_true", help="Sem janela (modo servidor)")
    args = parser.parse_args()

    print("🚗 FoxmoonDev — Car Tracking")
    print("=" * 40)

    run(
        source       = args.source,
        show_heatmap = args.heatmap,
        show_window  = not args.no_window,
    )

    print("\n✅ Concluído")
