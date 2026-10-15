# AgroVision AI

Weed detection for spot spraying in sugar beet and corn fields in Konya, Türkiye.
Field system: 16 m boom, 32 nozzles at 50 cm spacing, 6–8 cameras at 1 m height,
15 km/h. Field hardware: NVIDIA Jetson. Development: RTX 4070 laptop (8 GB).

**Current stage:** pipeline rehearsal on a Kaggle set (1300 static 512×512 images,
sesame crop + weeds, classes `0 = crop`, `1 = weed`). Real Konya field video
arrives in spring 2027.

## Files

| File | Purpose |
|---|---|
| `setup.md` | Install steps (CUDA PyTorch first, then the rest) and GPU verification |
| `train.py` | YOLO11n training; `WEIGHTS` / `RUN_NAME` at the top |
| `evaluate.py` | Go/no-go metric: strip-level weed recall vs. spray rate sweep, verdict |
| `infer_video.py` | Annotated video output + per-frame latency (mean, p95) |
| `extract_frames.py` | Field video → frames at 2 FPS, video ID in each filename |
| `split_dataset.py` | Train/val split by video/field, hard error on leakage |

## Pipeline order

Kaggle rehearsal (now):

1. `setup.md` — environment
2. `python train.py`
3. `python evaluate.py --weights runs/<RUN_NAME>/weights/best.pt`
4. `python infer_video.py <video> --weights ... --conf <threshold from step 3>`

Konya field data (2027):

1. `python extract_frames.py videos/ --fps 2 --out frames` — 2 FPS, **not** native fps
2. Label the frames (Roboflow or CVAT), export YOLO format
3. `python split_dataset.py --images frames --labels <export> --out dataset_konya --group-pattern "^([^_]+)_"`
   — split **by field/video**, never randomly by frame
4. Train at `imgsz=1280` with a smaller batch (see the comment block in `train.py`),
   pointing `DATA` at `dataset_konya/data.yaml`
5. Run the three variants below and compare them with `evaluate.py --imgsz 1280`

## Staged transfer learning

```
COCO (yolo11n.pt)  →  Kaggle  →  Konya
```

Each stage starts from the previous stage's `best.pt` (set `WEIGHTS` in `train.py`).
On Konya data, train and **compare** both:

- a) `yolo11n.pt` (COCO) → Konya
- b) Kaggle `best.pt` → Konya
- c) `yolo26n.pt` (COCO) → Konya — on Kaggle only weak evidence for it
  (+3 pts strip recall = 6 of 202 weeds, gone at spill tolerance 0.1)

The one with the lower spray rate at 98% **strip-level** weed recall wins. The Kaggle
stage (sesame, 512 px, other conditions) is not guaranteed to help, so it has to earn its place.

## Go / no-go decision rule

Standard mAP is not the decision metric. `evaluate.py` sweeps the confidence
threshold 0.05–0.95 and computes, per threshold, strip recall, box recall, weed
precision and **spray rate**: each frame is split into `NOZZLE_CELLS_PER_FRAME`
vertical strips (default 5 ≈ 2.6 m camera footprint / 0.5 m nozzle spacing), and a
strip is sprayed if any weed detection overlaps it.

- **Strip recall** (decision metric): fraction of ground-truth weeds that actually get
  sprayed, i.e. every strip the weed's box horizontally overlaps is sprayed, by any
  detection, however poorly localized. This is the spraying outcome.
- **Box recall** (reference only): fraction of weeds matched by a detection at IoU ≥ 0.5.
  This is detection quality, not spraying outcome.

> **At 98% strip-level weed recall, if the spray rate is above 65% (savings below 35%), the project stops.**

`SPILL_TOLERANCE` (default 0.0 = strict) lets a weed box spill a fraction of its width
into a neighbouring strip without that strip having to fire. Its value must come from
the **measured spray footprint of the real nozzle**, never from tuning until the result
passes. The script also prints the result at 0.1 once, as a sensitivity reference that
is not used for the verdict.

The script prints a PASS/FAIL block (with the box-recall result as a footnote) and writes
`runs/eval/<run>/recall_vs_spray.csv` and `.png` (strip recall solid, box recall dashed).
On the Kaggle rehearsal this verdict only validates the pipeline; the real go/no-go
comes from field data split by field.

## Non-negotiable rules

- Never split video frames randomly into train/val — always by source video / field.
- Extract frames at 2 FPS, not the native framerate.
- Real field data needs `imgsz=1280` (2–5 cm weeds).

## License note

Ultralytics YOLO is licensed under **AGPL-3.0**. That is fine for research and
internal development. If this ships as a product (e.g. closed firmware on a
commercial sprayer), either the whole product must comply with AGPL-3.0 or an
**Ultralytics Enterprise (commercial) license** is required.
