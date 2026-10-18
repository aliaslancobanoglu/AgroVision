"""
AgroVision AI — YOLO11n training.

Staged transfer learning:  COCO (yolo11n.pt) -> Kaggle -> Konya.
Each stage starts from the previous stage's best.pt, e.g.

    python train.py --weights runs/kaggle_yolo11n_640/weights/best.pt         --data dataset_konya/data.yaml --run-name konya_from_kaggle_1280         --imgsz 1280 --batch 4

A bare `python train.py` uses the defaults below (the Kaggle rehearsal run).
The resolved settings are printed at start.

Output lands in runs/<run-name>/. If that folder already exists this script
refuses to start instead of silently creating runs/<run-name>2, 3, ...
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Run configuration — defaults of the command-line options
# ----------------------------------------------------------------------------
WEIGHTS = 'yolo11n.pt'            # --weights: COCO start; later a prior run's best.pt
RUN_NAME = 'kaggle_yolo11n_640'   # --run-name: new name for every run
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
# --imgsz must go to 1280. Target weeds are 2-5 cm across; detecting them needs
# roughly 1-1.5 mm/pixel on the ground. At 640 px a ~2.6 m wide frame is
# ~4 mm/pixel, so a 2 cm weed is ~5 px — below what YOLO can reliably find.
# Note: even 1280 px over ~2.6 m is ~2 mm/pixel, still coarser than the
# 1-1.5 mm target. If the 2027 frames really cover 2.6 m each, plan for either
# narrower per-camera coverage (more cameras) or tiled inference.
#
# --batch must drop to fit 8 GB VRAM: 1280 px uses ~4x the activation memory of
# 640 px. Start around --batch 4 and adjust (--batch -1 lets Ultralytics
# auto-pick ~60% VRAM usage). If RAM becomes a problem, set CACHE = 'disk'
# or False.
#
#   python train.py --imgsz 1280 --batch 4 ...
# ============================================================================


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='YOLO training (staged transfer learning).')
    ap.add_argument('--data', default=DATA, help=f'dataset yaml (default {DATA})')
    ap.add_argument('--weights', default=WEIGHTS,
                    help=f"start weights: COCO .pt or a prior run's best.pt (default {WEIGHTS})")
    ap.add_argument('--run-name', default=RUN_NAME,
                    help=f'output folder runs/<run-name>, must not exist (default {RUN_NAME})')
    ap.add_argument('--imgsz', type=int, default=IMGSZ, help=f'1280 for field data (default {IMGSZ})')
    ap.add_argument('--batch', type=int, default=BATCH,
                    help=f'-1 = Ultralytics auto batch (default {BATCH})')
    ap.add_argument('--epochs', type=int, default=EPOCHS, help=f'default {EPOCHS}')
    ap.add_argument('--patience', type=int, default=PATIENCE, help=f'default {PATIENCE}')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)  # data.yaml uses a relative path (./dataset)

    run_dir = ROOT / PROJECT / args.run_name
    print(f"Data:     {args.data}")
    print(f"Weights:  {args.weights}")
    print(f"Run dir:  {run_dir}")
    print(f"imgsz {args.imgsz} | batch {args.batch} | epochs {args.epochs} | "
          f"patience {args.patience}")
    if run_dir.exists():
        raise SystemExit(
            f"runs/{args.run_name} already exists. Pass a different --run-name "
            f"(or delete the old run) — refusing to overwrite or auto-increment."
        )

    if torch.cuda.is_available():
        print(f"Device: cuda:{DEVICE} — {torch.cuda.get_device_name(DEVICE)} "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
    else:
        raise SystemExit("CUDA not available — see setup.md troubleshooting. "
                         "Refusing to train on CPU.")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=DEVICE,
        amp=True,
        workers=WORKERS,
        seed=SEED,
        cache=CACHE,
        project=str(ROOT / PROJECT),
        name=args.run_name,
        exist_ok=False,
    )


# Mandatory on Windows: dataloader workers are spawned processes that re-import
# this module. Without this guard, workers > 0 spawns processes recursively.
if __name__ == '__main__':
    main()
