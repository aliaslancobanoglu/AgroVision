"""
AgroVision AI — crop-stratified train/val split BY FIELD, never by frame.

CRITICAL CORRECTNESS REQUIREMENT
--------------------------------
Frames from the same video (or the same field) are near-duplicates: same soil,
light, crop stage, weed species, camera angle — often overlapping ground.
If frames are split randomly, val contains near-copies of training images.
The model is then graded on data it has effectively memorised: val metrics
(including the recall-vs-spray-rate verdict in evaluate.py) come out inflated,
the project looks like a PASS offline, and the model fails in a field it has
never seen. The only honest validation is on fields absent from training.

Therefore:
  - Every frame belongs to a group = crop_field, parsed from its video ID.
  - Whole groups go to either train or val.
  - Every crop must have at least one field in train AND one in val. A crop
    with fewer than 2 fields is a HARD ERROR (collect more fields; do NOT fall
    back to a random frame split).
  - After writing, the output directories are re-scanned and a HARD ERROR is
    raised if any video ID or group appears in both splits, or if a crop is
    missing from train or val.

Note: the current Kaggle rehearsal split in dataset/ was random-by-image. That
is correct for independent static photos, but WRONG for video frames. Do not
reuse that approach for the 2027 Konya data.

Input: frames produced by extract_frames.py, named
    <video_id>__f<frame_index:06d>.<ext>
where video_id follows the naming convention (see crops.py):
    <crop>_<field>_<YYYYMMDD>_<cam>        e.g. beet_F01_20270415_cam3
plus YOLO label .txt files with the same stems (from Roboflow/CVAT export).
Images without a label file are kept as background (negative) images.
An unknown crop or a malformed video ID is a hard error.

Usage
-----
    python split_dataset.py --images frames --labels labels_export --out dataset_konya
    python split_dataset.py ... --val-frac 0.2
    python split_dataset.py ... --val-groups beet_F03 corn_F07    # pin held-out fields
    python split_dataset.py --verify-only --out dataset_konya     # leakage check only

--val-frac is applied per crop (fraction of that crop's FRAMES, in whole fields).
The output folder gets its own data.yaml (classes wheat, beet, corn, weed unless
--names says otherwise); the project-root data.yaml (Kaggle) is never touched.
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from crops import CLASS_NAMES, parse_video_id, video_id_of_name

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class LeakageError(RuntimeError):
    pass


def video_id_of(path: Path) -> str:
    return video_id_of_name(path.name)


def group_of(video_id: str) -> str:
    return parse_video_id(video_id).group


def check_no_leakage(train_imgs, val_imgs):
    """Raise LeakageError if any video ID or group is present in both splits, or if a
    crop is missing from train or val."""
    tv = {video_id_of(p) for p in train_imgs}
    vv = {video_id_of(p) for p in val_imgs}
    both_v = tv & vv
    if both_v:
        raise LeakageError(f'Video IDs present in BOTH train and val: {sorted(both_v)}')
    tg = {group_of(v) for v in tv}
    vg = {group_of(v) for v in vv}
    both_g = tg & vg
    if both_g:
        raise LeakageError(f'Groups present in BOTH train and val: {sorted(both_g)}')
    tc = {parse_video_id(v).crop for v in tv}
    vc = {parse_video_id(v).crop for v in vv}
    if tc != vc:
        raise LeakageError(f'Every crop needs at least one field in train AND val. '
                           f'Only in train: {sorted(tc - vc)}; only in val: {sorted(vc - tc)}')
    return tv, vv, tg, vg


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.is_dir() else []


def verify(out: Path):
    train = list_images(out / 'images' / 'train')
    val = list_images(out / 'images' / 'val')
    if not train or not val:
        raise SystemExit(f'{out}: images/train or images/val is empty or missing.')
    tv, vv, tg, vg = check_no_leakage(train, val)
    print(f'OK — no leakage. train: {len(train)} frames / {len(tv)} videos / {len(tg)} fields | '
          f'val: {len(val)} frames / {len(vv)} videos / {len(vg)} fields')
    for crop in sorted({parse_video_id(v).crop for v in tv}):
        print(f'  {crop}: train fields {sorted(g for g in tg if g.startswith(crop + "_"))} | '
              f'val fields {sorted(g for g in vg if g.startswith(crop + "_"))}')


def choose_val_groups(by_group, val_frac, seed, pinned):
    """Pick val groups per crop. Hard error unless every crop keeps >=1 group in each split."""
    by_crop = defaultdict(list)
    for g in sorted(by_group):
        by_crop[g.split('_', 1)[0]].append(g)

    for crop, gs in sorted(by_crop.items()):
        if len(gs) < 2:
            raise SystemExit(f"Crop '{crop}' has only {len(gs)} field(s) ({gs}) — a crop-stratified "
                             f"split needs at least 2 fields per crop (one for train, one for val). "
                             f"Collect more fields; do NOT fall back to a random frame split.")

    if pinned:
        unknown = set(pinned) - set(by_group)
        if unknown:
            raise SystemExit(f'--val-groups not found: {sorted(unknown)}. Available: {sorted(by_group)}')
        val_groups = set(pinned)
        for crop, gs in sorted(by_crop.items()):
            n_val = sum(g in val_groups for g in gs)
            if n_val == 0 or n_val == len(gs):
                raise SystemExit(f"--val-groups leaves crop '{crop}' with {len(gs) - n_val} field(s) "
                                 f"in train and {n_val} in val — each crop needs at least one of each.")
        return val_groups

    rng = random.Random(seed)
    val_groups = set()
    for crop, gs in sorted(by_crop.items()):
        order = gs[:]
        rng.shuffle(order)
        target = val_frac * sum(len(by_group[g]) for g in gs)
        chosen, n_val = [], 0
        for g in order:
            if len(chosen) >= len(gs) - 1:
                break
            if chosen and n_val >= target:
                break
            chosen.append(g)
            n_val += len(by_group[g])
        val_groups.update(chosen)
    return val_groups


def main():
    ap = argparse.ArgumentParser(description='Split video frames into train/val by crop_field.')
    ap.add_argument('--images', default='frames', help='flat folder of extracted frames')
    ap.add_argument('--labels', default=None, help='folder of YOLO .txt labels (default: --images)')
    ap.add_argument('--out', default='dataset_konya', help='output dataset root (must be new/empty)')
    ap.add_argument('--val-frac', type=float, default=0.2,
                    help="target fraction of each crop's FRAMES in val (whole fields)")
    ap.add_argument('--val-groups', nargs='*', default=None,
                    help='explicit crop_field groups for val, e.g. beet_F03')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--move', action='store_true', help='move files instead of copying')
    ap.add_argument('--names', default=','.join(CLASS_NAMES),
                    help='class names for the output data.yaml')
    ap.add_argument('--verify-only', action='store_true', help='only check --out for leakage')
    args = ap.parse_args()

    out = Path(args.out)
    if args.verify_only:
        verify(out)
        return

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else img_dir
    imgs = list_images(img_dir)
    if not imgs:
        raise SystemExit(f'No images in {img_dir}')
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        if (out / sub).exists() and any((out / sub).iterdir()):
            raise SystemExit(f'{out / sub} is not empty — refusing to mix with an old split. '
                             f'Choose a new --out.')

    # ---- group frames (hard error on unknown crop / malformed video ID) ----
    by_group = defaultdict(list)
    for p in imgs:
        by_group[group_of(video_id_of(p))].append(p)
    groups = sorted(by_group)

    # ---- choose val groups -------------------------------------------------
    val_groups = choose_val_groups(by_group, args.val_frac, args.seed, args.val_groups)

    # ---- write -------------------------------------------------------------
    op = shutil.move if args.move else shutil.copy2
    manifest, n_bg = [], 0
    for g in groups:
        split = 'val' if g in val_groups else 'train'
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for p in by_group[g]:
            op(str(p), str(out / 'images' / split / p.name))
            lbl = lbl_dir / (p.stem + '.txt')
            if lbl.exists():
                op(str(lbl), str(out / 'labels' / split / lbl.name))
            else:
                n_bg += 1
            vid = video_id_of(p)
            manifest.append((p.name, vid, parse_video_id(vid).crop, g, split))

    with open(out / 'split_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'video_id', 'crop', 'group', 'split'])
        w.writerows(manifest)

    names = [n.strip() for n in args.names.split(',')]
    (out / 'data.yaml').write_text(
        f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n'
        f'nc: {len(names)}\nnames: {names}\n', encoding='utf-8')

    # ---- hard leakage check on what is actually on disk --------------------
    verify(out)
    print(f'val groups: {sorted(val_groups)}')
    if n_bg:
        print(f'Note: {n_bg} image(s) had no label file and were kept as background images.')
    print(f'Manifest: {out / "split_manifest.csv"}\nData yaml: {out / "data.yaml"}')


if __name__ == '__main__':
    main()
