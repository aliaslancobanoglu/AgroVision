"""
AgroVision AI — run the detector on a video and write an annotated copy.

Reports mean and p95 per-frame latency in ms:
  - model   : Ultralytics' own preprocess + inference + postprocess timing
  - e2e     : wall-clock time of the whole predict() call per frame
The first WARMUP_FRAMES frames are excluded (CUDA init / cuDNN autotune).

Laptop numbers are only indicative — the field target is a Jetson, where the
model should be exported to TensorRT and re-timed on the device.

Usage
-----
    python infer_video.py path/to/video.mp4
    python infer_video.py video.mp4 --weights runs/<RUN>/weights/best.pt --conf 0.25 --imgsz 1280
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = 'runs/kaggle_yolo11n_640/weights/best.pt'
WARMUP_FRAMES = 10


def main():
    ap = argparse.ArgumentParser(description='Annotate a video with weed detections.')
    ap.add_argument('video')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--out', default=None, help='default: runs/video/<video stem>_annotated.mp4')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='use the threshold chosen by evaluate.py')
    ap.add_argument('--imgsz', type=int, default=640, help='1280 for real field data')
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Device: {'cuda:0 — ' + torch.cuda.get_device_name(0) if device == 0 else 'cpu'}")

    src = Path(args.video)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {src}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = Path(args.out) if args.out else Path('runs/video') / f'{src.stem}_annotated.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video writer for {out}')

    model = YOLO(args.weights)
    print(f'Input : {src} ({w}x{h} @ {fps:.2f} fps, ~{n_total} frames)')
    print(f'Output: {out}')

    model_ms, e2e_ms = [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                device=device, verbose=False)[0]
            if device == 0:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= WARMUP_FRAMES:
                e2e_ms.append((t1 - t0) * 1000)
                model_ms.append(sum(res.speed.values()))
            writer.write(res.plot())
            i += 1
            if i % 100 == 0:
                print(f'  {i} frames')
    finally:
        cap.release()
        writer.release()

    print(f'\nFrames processed: {i} (first {WARMUP_FRAMES} excluded from timing)')
    if not e2e_ms:
        print('Not enough frames for timing stats.')
        return
    for name, v in (('model', model_ms), ('e2e  ', e2e_ms)):
        v = np.asarray(v)
        print(f'{name} per-frame: mean {v.mean():6.2f} ms | p95 {np.percentile(v, 95):6.2f} ms')
    print(f'Throughput (e2e mean): {1000 / np.mean(e2e_ms):.1f} FPS')


if __name__ == '__main__':
    main()
