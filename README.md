# 🚗 Car Detection & Tracking

Pipeline completo de detecção e rastreamento de carros em imagens, vídeos e streams de câmera (DVR via RTSP).

> Desenvolvido por **FoxmoonDev**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_cu124-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Datasets

| # | Dataset | Finalidade |
|---|---------|------------|
| 1 | [Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) | Treinar o detector |
| 2 | [Cars Video Object Tracking](https://www.kaggle.com/datasets/trainingdatapro/cars-video-object-tracking) | Fine-tune + tracking em vídeo |

---

## Estrutura do Projeto

```
car-detection-tracking/
│
├── 1_prepare_dataset.py     Dataset 1 — CSV → formato YOLO
├── 2_train.py               Treina o modelo YOLOv8
├── 3_inference.py           Inferência em imagens
├── 4_tracking.py            Tracking em vídeo, RTSP ou webcam
├── 5_prepare_dataset2.py    Dataset 2 — combina com Dataset 1
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
│
├── data/                    Dataset 1 (não versionado)
├── data2/                   Dataset 2 (não versionado)
│
├── car_detection_dataset/   Gerado automaticamente
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
│
├── runs/
│   └── car_detector/
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
├── predictions/             Imagens anotadas
└── tracking_output/         Vídeo, CSV, heatmap, trajetórias
```

---

## Instalação

```bash
git clone https://github.com/FoxmoonDev/car-detection-tracking.git
cd car-detection-tracking

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Verificar GPU:
```bash
python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available())"
```

---

## Uso

### Fase 1 — Treinar o detector

```bash
# Extraia o Dataset 1 em ./data/
python 1_prepare_dataset.py
python 2_train.py
python 3_inference.py
```

### Fase 2 — Fine-tune com vídeos

```bash
# Extraia o Dataset 2 em ./data2/
python 5_prepare_dataset2.py
python 2_train.py
```

### Fase 3 — Tracking

```bash
# Arquivo de vídeo
python 4_tracking.py --source video.mp4

# Stream RTSP (DVR / câmera IP)
python 4_tracking.py --source rtsp://admin:senha@192.168.1.100:554/...

# Webcam
python 4_tracking.py --source 0

# Com heatmap
python 4_tracking.py --source video.mp4 --heatmap

# Sem janela (servidor)
python 4_tracking.py --source video.mp4 --no-window
```

### Inferência

```bash
python 3_inference.py
python 3_inference.py --source imagem.jpg
python 3_inference.py --source pasta/
python 3_inference.py --export
```

---

## DVR via RTSP

Teste no VLC primeiro:
```
vlc rtsp://usuario:senha@IP:554/...
```

| Fabricante | URL |
|------------|-----|
| Hikvision | `rtsp://user:senha@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:senha@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Intelbras | `rtsp://user:senha@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Genérico | `rtsp://user:senha@IP:554/live/ch0` |

---

## Resultados

| Métrica | Valor |
|---------|-------|
| mAP50 | ~0.97 |
| mAP50-95 | ~0.65 |
| Inferência | ~7ms/frame (RTX 4060) |

---

## Saídas do Tracking

| Arquivo | Conteúdo |
|---------|----------|
| `*_tracked.mp4` | Vídeo com IDs, trilhas e velocidades |
| `*_data.csv` | Dados por frame (id, bbox, velocidade) |
| `*_heatmap.jpg` | Mapa de calor das trajetórias |
| `*_trajectories.png` | Gráfico das rotas por ID |

---

## Calibração de Velocidade

Em `4_tracking.py`:
```python
PIXELS_PER_METER = 15.0
```
Meça quantos pixels equivalem a 1 metro na sua câmera. Exemplo: carro (~4m) ocupando 60px → `60 / 4 = 15`.

---

## GPU Recomendada

| GPU | Modelo | Batch |
|-----|--------|-------|
| RTX 4060 8GB | yolov8s | 16 |
| RTX 3060 12GB | yolov8m | 16 |
| RTX 4090 24GB | yolov8l | 32 |
| CPU | yolov8n | 4 |

---

## Referências

- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [ByteTrack](https://arxiv.org/abs/2110.06864)

---

## Licença

MIT License — veja [LICENSE](LICENSE)

---

<p align="center">Made with ❤️ by <strong>FoxmoonDev</strong></p>
