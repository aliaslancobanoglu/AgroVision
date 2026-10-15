"""
AgroVision AI — YOLO11n training.

Staged transfer learning:  COCO (yolo11n.pt) -> Kaggle -> Konya.
Each stage starts from the previous stage's best.pt, e.g.

    WEIGHTS  = 'runs/kaggle_yolo11n_640/weights/best.pt'
    RUN_NAME = 'konya_from_kaggle_1280'

Output lands in runs/<RUN_NAME>/. If that folder already exists this script
refuses to start instead of silently creating runs/<RUN_NAME>2, 3, ...
"""

import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------------
WEIGHTS = 'yolo11n.pt'            # COCO start; later: a prior run's best.pt
RUN_NAME = 'kaggle_yolo11n_640'   # change for every new run
DATA = 'data.yaml'
PROJECT = 'runs'

IMGSZ = 640
BATCH = 16
EPOCHS = 100
PATIENCE = 20
DEVICE = 0
WORKERS = 2
SEED = 0
CACHE = 'ram'

# ============================================================================
# REAL FIELD DATA (Konya, spring 2027) — READ BEFORE TRAINING ON IT
# ----------------------------------------------------------------------------
# IMGSZ must go to 1280. Target weeds are 2-5 cm across; detecting them needs
# roughly 1-1.5 mm/pixel on the ground. At 640 px a ~2.6 m wide frame is
# ~4 mm/pixel, so a 2 cm weed is ~5 px — below what YOLO can reliably find.
# Note: even 1280 px over ~2.6 m is ~2 mm/pixel, still coarser than the
# 1-1.5 mm target. If the 2027 frames really cover 2.6 m each, plan for either
# narrower per-camera coverage (more cameras) or tiled inference.
#
# BATCH must drop to fit 8 GB VRAM: 1280 px uses ~4x the activation memory of
# 640 px. Start around BATCH = 4 and adjust (BATCH = -1 lets Ultralytics
# auto-pick ~60% VRAM usage). If RAM becomes a problem, set CACHE = 'disk'
# or False.
#
#   IMGSZ = 1280
#   BATCH = 4
# ============================================================================


def main():
    os.chdir(ROOT)  # data.yaml uses a relative path (./dataset)

    if (ROOT / PROJECT / RUN_NAME).exists():
        raise SystemExit(
            f"runs/{RUN_NAME} already exists. Change RUN_NAME at the top of "
            f"train.py (or delete the old run) — refusing to overwrite or auto-increment."
        )

    if torch.cuda.is_available():
        print(f"Device: cuda:{DEVICE} — {torch.cuda.get_device_name(DEVICE)} "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
    else:
        raise SystemExit("CUDA not available — see setup.md troubleshooting. "
                         "Refusing to train on CPU.")
    print(f"Weights: {WEIGHTS}")
    print(f"Run dir: {ROOT / PROJECT / RUN_NAME}")

    model = YOLO(WEIGHTS)
    model.train(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        amp=True,
        workers=WORKERS,
        seed=SEED,
        cache=CACHE,
        project=str(ROOT / PROJECT),
        name=RUN_NAME,
        exist_ok=False,
    )


# Mandatory on Windows: dataloader workers are spawned processes that re-import
# this module. Without this guard, workers > 0 spawns processes recursively.
if __name__ == '__main__':
    main()
