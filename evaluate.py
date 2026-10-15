"""
AgroVision AI — go/no-go evaluation: weed recall vs. spray rate.

Standard mAP is NOT the decision metric. What matters for spot spraying is the
trade-off between MISSING weeds (low recall -> weeds survive) and spraying
UNNECESSARILY (high spray rate -> little herbicide saved).

Decision rule (see the project notes / README.md):
    At 98% STRIP-LEVEL weed recall, if the sprayed-area fraction is above 65%
    (i.e. savings below 35%), the project stops.

Two kinds of recall — why the verdict uses strip recall
-------------------------------------------------------
- Box recall (weed_recall) asks "did the detector draw a box with IoU >= 0.5
  on this weed?". That measures DETECTION quality.
- Strip recall (strip_recall) asks "was this weed actually sprayed?". A weed
  is treated when every nozzle strip its GT box horizontally overlaps is
  sprayed — by any detection, however poorly localized. A box at IoU 0.3, a
  box covering two neighbouring weeds, or a box hitting the right strip
  because of a different weed all still kill the weed. That measures the
  SPRAYING OUTCOME, which is what the go/no-go rule is about.
Strip recall is NOT always >= box recall. The "every strip" rule is strict: a
GT box whose edge (often just a leaf tip) reaches a few pixels into a
neighbouring strip counts as untreated unless that strip fires too, even when
the detection matched at IoU >= 0.5. On the Kaggle val set at conf 0.05 this
cost 9 IoU-matched weeds (1-41 px spill, mostly < 6% of box width) while 8
poorly-localized-but-sprayed weeds were gained. Box recall is still reported
(CSV, dashed curve, verdict footnote) as a model-quality reference.

Spill tolerance (SPILL_TOLERANCE)
---------------------------------
A strip counts as "needed" for a GT box only if MORE than SPILL_TOLERANCE of
the box's width lies in that strip; the strip with the largest overlap is
always needed, so no box is ever treated without spraying. 0.0 (default) is
the strict rule above, and the VERDICT ALWAYS USES THE CONSTANT AS SET.

The real value must come from the MEASURED spray footprint of the actual
nozzle: how far its fan reaches beyond the strip edge on the ground, expressed
as a fraction of a typical weed box width. It must NEVER be tuned until the
results pass — that would turn the go/no-go rule into a self-fulfilling one.
The script also prints the result at SPILL_TOLERANCE_REFERENCE (0.1) once,
labelled "reference only", purely to show how sensitive the verdict is to it.

Method
------
1. Run inference once on the val split in data.yaml at the lowest swept
   confidence (0.05). Filtering those detections by a higher confidence gives
   exactly the detections the model would produce at that higher threshold
   (NMS is greedy by score, so removing low-score boxes never changes which
   high-score boxes survive; the same holds for NMS-free end-to-end heads such
   as YOLO26, whose top-k selection is also by score), so one inference pass
   serves the whole sweep.
2. For conf in 0.05, 0.10, ..., 0.95:
     - strip recall   = GT weed boxes whose overlapped strips are ALL sprayed
                        / all GT weed boxes (class 1).
     - weed recall    = matched GT weed boxes / all GT weed boxes (class 1),
                        greedy one-to-one matching, highest confidence first,
                        IoU >= 0.5. (Box recall.)
     - weed precision = matched weed detections / all weed detections.
     - spray rate     = sprayed strips / total strips over the whole val set.
   Images with zero GT weeds contribute 0 to recall denominators (no
   per-image division anywhere).

Spray-rate model (nozzle cells)
-------------------------------
Each image's width is split into NOZZLE_CELLS_PER_FRAME equal vertical strips.
A strip is "sprayed" if any weed detection at that threshold overlaps it
horizontally. Each strip stands for one nozzle's band on the ground.

Real system: a 16 m boom carries 32 nozzles at 50 cm spacing, seen by 6-8
cameras, so each camera covers ~2.6 m of boom width, i.e. ~5 nozzle cells per
frame. Once the real camera's ground coverage is measured, ONLY the constant
below changes (cells = frame ground width / 0.5 m).

The real system also has a time dimension: a nozzle stays open for a duration
as the boom moves forward (15 km/h), so the along-track extent matters too.
Per-frame vertical strips are the correct static proxy for that — they measure
what fraction of the nozzle bands must fire for the area this frame covers.

For reference the script also reports the "oracle" spray rate: strips hit by
ground-truth weed boxes. No detector can save more than the oracle does at
100% strip recall; if the oracle itself is above 65%, the field (or the strip
width) makes the project fail regardless of model quality.

Usage
-----
    python evaluate.py                                  # default weights
    python evaluate.py --weights runs/<RUN>/weights/best.pt
    python evaluate.py --weights ... --imgsz 1280       # real field data
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NOZZLE_CELLS_PER_FRAME = 5     # ~2.6 m frame / 0.5 m nozzle spacing
WEED_CLASS = 1
IOU_MATCH = 0.5
CONF_THRESHOLDS = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)

# Fraction of a GT box's width that may spill into a neighbouring strip without
# that strip having to fire (see "Spill tolerance" in the header). 0.0 = strict.
# Set ONLY from the measured spray footprint of the real nozzle.
SPILL_TOLERANCE = 0.0
SPILL_TOLERANCE_REFERENCE = 0.1   # printed once for sensitivity, never used for the verdict

TARGET_RECALL = 0.98
SAVINGS_FLOOR = 0.35           # spray rate must be <= 1 - 0.35 = 0.65

DEFAULT_WEIGHTS = 'runs/kaggle_yolo11n_640/weights/best.pt'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def resolve_val_dirs(data_yaml: Path):
    """Return (images_dir, labels_dir) for the val split in data.yaml."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    base = Path(cfg.get('path', '.'))
    if not base.is_absolute():
        # Ultralytics resolves relative 'path' against the CWD; we chdir to ROOT.
        base = (Path.cwd() / base).resolve()
    val = cfg['val']
    if not isinstance(val, str):
        raise ValueError("evaluate.py expects 'val' in data.yaml to be a single directory.")
    img_dir = (base / val).resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Val image dir not found: {img_dir}")
    # YOLO convention: .../images/... -> .../labels/...
    parts = list(img_dir.parts)
    if 'images' not in parts:
        raise ValueError(f"Cannot derive labels dir from {img_dir} (no 'images' component).")
    idx = len(parts) - 1 - parts[::-1].index('images')
    parts[idx] = 'labels'
    return img_dir, Path(*parts)


def load_gt_weeds(label_path: Path, w: int, h: int) -> np.ndarray:
    """GT weed boxes as (N, 4) xyxy pixels. Missing/empty label file -> no boxes."""
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [ln.split() for ln in label_path.read_text().splitlines() if ln.strip()]
    boxes = []
    for r in rows:
        if int(float(r[0])) != WEED_CLASS:
            continue
        xc, yc, bw, bh = (float(v) for v in r[1:5])
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(boxes, dtype=float).reshape(-1, 4)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def count_matches(pred: np.ndarray, pred_conf: np.ndarray, gt: np.ndarray) -> int:
    """Greedy one-to-one matching, highest-confidence prediction first."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    iou = box_iou(pred, gt)
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for i in np.argsort(-pred_conf):
        cand = np.where(~used & (iou[i] >= IOU_MATCH))[0]
        if len(cand):
            used[cand[np.argmax(iou[i, cand])]] = True
            tp += 1
    return tp


def strip_hits(boxes: np.ndarray, width: int, n_cells: int) -> np.ndarray:
    """(N, n_cells) bool: which of the n_cells vertical strips each box horizontally overlaps."""
    if len(boxes) == 0:
        return np.zeros((0, n_cells), dtype=bool)
    edges = np.linspace(0, width, n_cells + 1)
    left, right = edges[:-1], edges[1:]
    x1 = np.clip(boxes[:, 0], 0, width)[:, None]
    x2 = np.clip(boxes[:, 2], 0, width)[:, None]
    return (x1 < right[None, :]) & (x2 > left[None, :])


def sprayed_strips(boxes: np.ndarray, width: int, n_cells: int) -> int:
    """Number of the n_cells vertical strips that any box horizontally overlaps."""
    return int(strip_hits(boxes, width, n_cells).any(axis=0).sum())


def needed_strips(gt: np.ndarray, width: int, n_cells: int, spill_tol: float) -> np.ndarray:
    """(N, n_cells) bool: strips that must fire for each GT box to count as treated.

    A strip is needed if MORE than spill_tol of the box's width lies in it. The
    strip with the largest overlap is always needed, and a degenerate box that
    overlaps no strip (zero width on a strip edge) needs the strip holding its
    centre — so no box is ever treated without any spraying. With spill_tol =
    0.0 this is exactly strip_hits (every overlapped strip is needed) for every
    box of non-zero width.
    """
    hits = strip_hits(gt, width, n_cells)
    if len(gt) == 0:
        return hits
    need = hits.copy()
    if spill_tol > 0:
        edges = np.linspace(0, width, n_cells + 1)
        x1 = np.clip(gt[:, 0], 0, width)[:, None]
        x2 = np.clip(gt[:, 2], 0, width)[:, None]
        overlap = np.clip(np.minimum(x2, edges[None, 1:]) - np.maximum(x1, edges[None, :-1]),
                          0, None)
        box_w = (x2 - x1)[:, 0]
        pos = box_w > 0
        need[pos] = overlap[pos] / box_w[pos, None] > spill_tol
        rows = np.where(hits.any(axis=1))[0]
        need[rows, overlap[rows].argmax(axis=1)] = True
        need &= hits
    empty = ~need.any(axis=1)
    if empty.any():
        cx = np.clip((gt[empty, 0] + gt[empty, 2]) / 2, 0, width)
        need[np.where(empty)[0],
             np.minimum((cx / width * n_cells).astype(int), n_cells - 1)] = True
    return need


def count_treated(gt: np.ndarray, sprayed: np.ndarray, width: int, n_cells: int,
                  spill_tol: float = 0.0) -> int:
    """GT boxes for which every needed strip is in the `sprayed` (n_cells,) mask."""
    need = needed_strips(gt, width, n_cells, spill_tol)
    return int((~(need & ~sprayed[None, :])).all(axis=1).sum())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_inference(weights, img_paths, imgsz, device):
    from ultralytics import YOLO
    model = YOLO(weights)
    per_image = []
    speeds = []
    for res in model.predict(source=[str(p) for p in img_paths], imgsz=imgsz,
                             conf=float(CONF_THRESHOLDS.min()), device=device,
                             stream=True, verbose=False):
        speeds.append(res.speed)
        b = res.boxes
        cls = b.cls.cpu().numpy().astype(int)
        keep = cls == WEED_CLASS
        h, w = res.orig_shape
        per_image.append({
            'path': Path(res.path),
            'w': w, 'h': h,
            'xyxy': b.xyxy.cpu().numpy()[keep],
            'conf': b.conf.cpu().numpy()[keep],
        })
    # Median ms/image per stage (Ultralytics timings, PyTorch, batch 1, conf 0.05).
    # The median ignores the slow first-image warm-up.
    speed = {k: float(np.median([s[k] for s in speeds])) for k in speeds[0]}
    return per_image, speed


def sweep(per_image, labels_dir):
    for d in per_image:
        d['gt'] = load_gt_weeds(labels_dir / (d['path'].stem + '.txt'), d['w'], d['h'])

    n_gt = sum(len(d['gt']) for d in per_image)
    total_strips = len(per_image) * NOZZLE_CELLS_PER_FRAME
    oracle = sum(sprayed_strips(d['gt'], d['w'], NOZZLE_CELLS_PER_FRAME)
                 for d in per_image) / total_strips

    rows = []
    for t in CONF_THRESHOLDS:
        tp = n_pred = strips = treated = 0
        for d in per_image:
            m = d['conf'] >= t
            boxes, conf = d['xyxy'][m], d['conf'][m]
            n_pred += len(boxes)
            tp += count_matches(boxes, conf, d['gt'])
            sprayed = strip_hits(boxes, d['w'], NOZZLE_CELLS_PER_FRAME).any(axis=0)
            strips += int(sprayed.sum())
            treated += count_treated(d['gt'], sprayed, d['w'], NOZZLE_CELLS_PER_FRAME,
                                     SPILL_TOLERANCE)
        rows.append({
            'conf': float(t),
            'strip_recall': treated / n_gt if n_gt else float('nan'),
            'weed_recall': tp / n_gt if n_gt else float('nan'),
            'weed_precision': tp / n_pred if n_pred else float('nan'),
            'spray_rate': strips / total_strips,
            'savings': 1 - strips / total_strips,
            'tp': tp, 'fp': n_pred - tp, 'fn': n_gt - tp,
            'treated_weeds': treated,
            'sprayed_strips': strips,
        })
    return pd.DataFrame(rows), n_gt, total_strips, oracle


def strip_recall_at(per_image, n_gt, spill_tol):
    """Strip recall per CONF_THRESHOLDS for another spill tolerance (needs sweep() first)."""
    out = []
    for t in CONF_THRESHOLDS:
        treated = 0
        for d in per_image:
            sprayed = strip_hits(d['xyxy'][d['conf'] >= t], d['w'],
                                 NOZZLE_CELLS_PER_FRAME).any(axis=0)
            treated += count_treated(d['gt'], sprayed, d['w'], NOZZLE_CELLS_PER_FRAME, spill_tol)
        out.append(treated / n_gt)
    return np.array(out)


def print_spill_reference(df, per_image, n_gt):
    """Sensitivity to the spill tolerance — reference only, not used for the verdict."""
    tol = SPILL_TOLERANCE_REFERENCE
    rec = strip_recall_at(per_image, n_gt, tol)
    ok = rec >= TARGET_RECALL
    print(f'  [REFERENCE ONLY, not used for the verdict] spill tolerance {tol:.2f}:')
    print(f'      max strip recall {rec.max():.2%} '
          f'(verdict tolerance {SPILL_TOLERANCE:.2f}: {df["strip_recall"].max():.2%})')
    if ok.any():
        sr = df['spray_rate'].to_numpy()[ok]
        print(f'      {TARGET_RECALL:.0%} strip recall reached, lowest spray rate {sr.min():.2%}')
    else:
        print(f'      {TARGET_RECALL:.0%} strip recall not reached')
    print('=' * 64)


def plot(df, oracle, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid, series = '#1f2328', '#6e7781', '#d0d7de', '#2a6fdb'
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    ax.plot(df['strip_recall'], df['spray_rate'], color=series, lw=2,
            marker='o', ms=5, label='Strip recall — decision metric (conf 0.05–0.95)')
    ax.plot(df['weed_recall'], df['spray_rate'], color=series, lw=1.2, ls='--',
            alpha=0.6, label='Box recall, IoU ≥ 0.5 — reference')
    # Label a few thresholds only, not every point.
    for _, r in df[df['conf'].isin([0.05, 0.25, 0.5, 0.75, 0.95])].iterrows():
        if np.isfinite(r['strip_recall']):
            ax.annotate(f"{r['conf']:.2f}", (r['strip_recall'], r['spray_rate']),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=8, color=muted)
    ax.axvline(TARGET_RECALL, color=muted, lw=1, ls='--')
    ax.axhline(1 - SAVINGS_FLOOR, color=muted, lw=1, ls='--')
    ax.axhline(oracle, color=muted, lw=1, ls=':', label=f'Oracle (GT boxes) = {oracle:.1%}')
    ax.text(TARGET_RECALL, 1.01, f' {TARGET_RECALL:.0%} recall', color=ink, fontsize=8,
            ha='right', va='bottom', transform=ax.get_xaxis_transform())
    ax.text(0.005, 1 - SAVINGS_FLOOR, f'max spray rate {1 - SAVINGS_FLOOR:.0%}',
            color=ink, fontsize=8, va='bottom', transform=ax.get_yaxis_transform())
    ax.fill_between([TARGET_RECALL, 1.0], 0, 1 - SAVINGS_FLOOR, color=series, alpha=0.06, lw=0)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Weed recall (solid: strip-level, dashed: box IoU ≥ 0.5)', color=ink)
    ax.set_ylabel(f'Spray rate ({NOZZLE_CELLS_PER_FRAME} nozzle strips / frame)', color=ink)
    ax.set_title('Strip-level weed recall vs. spray rate — PASS region shaded',
                 color=ink, fontsize=11)
    ax.grid(color=grid, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def best_at_target(df, recall_col):
    """Lowest-spray-rate row reaching TARGET_RECALL on recall_col, or None."""
    ok = df[df[recall_col] >= TARGET_RECALL]
    return None if ok.empty else ok.loc[ok['spray_rate'].idxmin()]


def verdict(df, oracle):
    print('\n' + '=' * 64)
    print(f'  GO / NO-GO VERDICT  (nozzle cells per frame = {NOZZLE_CELLS_PER_FRAME})')
    print('  Decision metric: STRIP-LEVEL weed recall')
    print('=' * 64)
    row = best_at_target(df, 'strip_recall')
    if row is None:
        print(f'  No threshold in {CONF_THRESHOLDS.min():.2f}-{CONF_THRESHOLDS.max():.2f} '
              f'reaches {TARGET_RECALL:.0%} strip recall (max = {df["strip_recall"].max():.2%}).')
        print(f'  Oracle spray rate (GT boxes)         : {oracle:.2%}')
        print('  RESULT: FAIL — target strip recall not reachable with this model.')
        passed = False
    else:
        passed = row['savings'] >= SAVINGS_FLOOR
        print(f'  Threshold achieving >= {TARGET_RECALL:.0%} strip recall: conf = {row["conf"]:.2f}')
        print(f'  Strip recall                         : {row["strip_recall"]:.2%}')
        print(f'  Box recall (IoU >= 0.5)              : {row["weed_recall"]:.2%}')
        print(f'  Weed precision                       : {row["weed_precision"]:.2%}')
        print(f'  Spray rate                           : {row["spray_rate"]:.2%}')
        print(f'  Savings (1 - spray rate)             : {row["savings"]:.2%}')
        print(f'  Savings floor                        : {SAVINGS_FLOOR:.0%}')
        print(f'  Oracle spray rate (GT boxes)         : {oracle:.2%}')
        print(f'  RESULT: {"PASS" if passed else "FAIL"}')

    # Reference only — does NOT affect the verdict.
    print('-' * 64)
    ref = best_at_target(df, 'weed_recall')
    if ref is None:
        print(f'  [reference] Box recall (IoU >= 0.5): {TARGET_RECALL:.0%} not reached '
              f'(max = {df["weed_recall"].max():.2%}).')
    else:
        ref_pass = ref['savings'] >= SAVINGS_FLOOR
        print(f'  [reference] Box recall (IoU >= 0.5): {TARGET_RECALL:.0%} reached at '
              f'conf = {ref["conf"]:.2f}, spray rate {ref["spray_rate"]:.2%} '
              f'-> would be {"PASS" if ref_pass else "FAIL"}')
    print('-' * 64)
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--data', default='data.yaml')
    ap.add_argument('--imgsz', type=int, default=640, help='1280 for real field data')
    ap.add_argument('--out', default=None, help='output dir (default runs/eval/<weights run name>)')
    args = ap.parse_args()

    os.chdir(ROOT)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Device: {'cuda:0 — ' + torch.cuda.get_device_name(0) if device == 0 else 'cpu'}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f'Weights not found: {weights}. Pass --weights runs/<RUN>/weights/best.pt')
    img_dir, labels_dir = resolve_val_dirs(Path(args.data))
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        raise FileNotFoundError(f'No images in {img_dir}')
    print(f'Val images: {len(img_paths)} in {img_dir}')

    run_name = weights.parent.parent.name if weights.parent.name == 'weights' else weights.stem
    out_dir = Path(args.out) if args.out else ROOT / 'runs' / 'eval' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image, speed = run_inference(str(weights), img_paths, args.imgsz, device)
    df, n_gt, total_strips, oracle = sweep(per_image, labels_dir)
    if n_gt == 0:
        raise RuntimeError('Val set contains no ground-truth weed boxes — recall is undefined.')

    csv_path, png_path = out_dir / 'recall_vs_spray.csv', out_dir / 'recall_vs_spray.png'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    plot(df, oracle, png_path)

    print(f'GT weed boxes: {n_gt} | total nozzle strips: {total_strips}')
    print('Speed (median ms/image, imgsz {}): preprocess {:.2f} | inference {:.2f} | '
          'postprocess {:.2f}'.format(args.imgsz, speed['preprocess'], speed['inference'],
                                      speed['postprocess']))
    with pd.option_context('display.width', 120):
        print(df[['conf', 'strip_recall', 'weed_recall', 'weed_precision', 'spray_rate', 'savings']]
              .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nCSV : {csv_path}\nPlot: {png_path}')
    verdict(df, oracle)
    print_spill_reference(df, per_image, n_gt)


if __name__ == '__main__':
    main()
