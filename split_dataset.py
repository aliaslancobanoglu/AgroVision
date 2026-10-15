"""
AgroVision AI — train/val split BY SOURCE VIDEO / FIELD, never by frame.

CRITICAL CORRECTNESS REQUIREMENT
--------------------------------
Frames from the same video (or the same field) are near-duplicates: same soil,
light, crop stage, weed species, camera angle — often overlapping ground.
If frames are split randomly, val contains near-copies of training images.
The model is then graded on data it has effectively memorised: val metrics
(including the recall-vs-spray-rate verdict in evaluate.py) come out inflated,
the project looks like a PASS offline, and the model fails in a field it has
never seen. The only honest validation is on videos/fields absent from training.

Therefore:
  - Every frame is assigned to a group (its video ID, or its field ID).
  - Whole groups go to either train or val.
  - After writing, the output directories are re-scanned and a HARD ERROR is
    raised if any video ID (or group) appears in both splits.

Note: the current Kaggle rehearsal split in dataset/ was random-by-image. That
is correct for independent static photos, but WRONG for video frames. Do not
reuse that approach for the 2027 Konya data.

Input: frames produced by extract_frames.py, named
    <video_id>__f<frame_index:06d>.<ext>
plus YOLO label .txt files with the same stems (from Roboflow/CVAT export).
Images without a label file are kept as background (negative) images.

Grouping (default = video ID). the project notes asks for a FIELD-level split where
possible — with names like FIELD01_20270415_cam3, use:
    --group-pattern "^([^_]+)_"
so every video from FIELD01 lands on the same side.

Usage
-----
    python split_dataset.py --images frames --labels labels_export --out dataset_konya
    python split_dataset.py ... --group-pattern "^([^_]+)_" --val-frac 0.2
    python split_dataset.py ... --val-groups FIELD03 FIELD07      # pin held-out fields
    python split_dataset.py --verify-only --out dataset_konya     # leakage check only

The output folder gets its own data.yaml; the project-root data.yaml (Kaggle)
is never touched.
"""

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEP = '__'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class LeakageError(RuntimeError):
    pass


def video_id_of(path: Path) -> str:
    if SEP not in path.stem:
        raise ValueError(f"'{path.name}' has no '{SEP}' separator — cannot tell which video it "
                         f"came from. Frames must be named by extract_frames.py.")
    return path.stem.split(SEP)[0]


def group_of(video_id: str, pattern: re.Pattern) -> str:
    m = pattern.search(video_id)
    if not m:
        raise ValueError(f"--group-pattern did not match video ID '{video_id}'")
    return m.group(1) if m.groups() else m.group(0)


def check_no_leakage(train_imgs, val_imgs, pattern: re.Pattern):
    """Raise LeakageError if any video ID or group is present in both splits."""
    tv = {video_id_of(p) for p in train_imgs}
    vv = {video_id_of(p) for p in val_imgs}
    both_v = tv & vv
    if both_v:
        raise LeakageError(f'Video IDs present in BOTH train and val: {sorted(both_v)}')
    tg = {group_of(v, pattern) for v in tv}
    vg = {group_of(v, pattern) for v in vv}
    both_g = tg & vg
    if both_g:
        raise LeakageError(f'Groups present in BOTH train and val: {sorted(both_g)}')
    return tv, vv, tg, vg


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.is_dir() else []


def verify(out: Path, pattern: re.Pattern):
    train = list_images(out / 'images' / 'train')
    val = list_images(out / 'images' / 'val')
    if not train or not val:
        raise SystemExit(f'{out}: images/train or images/val is empty or missing.')
    tv, vv, tg, vg = check_no_leakage(train, val, pattern)
    print(f'OK — no leakage. train: {len(train)} frames / {len(tv)} videos / {len(tg)} groups | '
          f'val: {len(val)} frames / {len(vv)} videos / {len(vg)} groups')


def main():
    ap = argparse.ArgumentParser(description='Split video frames into train/val by video/field.')
    ap.add_argument('--images', default='frames', help='flat folder of extracted frames')
    ap.add_argument('--labels', default=None, help='folder of YOLO .txt labels (default: --images)')
    ap.add_argument('--out', default='dataset_konya', help='output dataset root (must be new/empty)')
    ap.add_argument('--val-frac', type=float, default=0.2, help='target fraction of FRAMES in val')
    ap.add_argument('--group-pattern', default=r'^(.*)$',
                    help='regex on video ID; group 1 = split unit (default: whole video ID)')
    ap.add_argument('--val-groups', nargs='*', default=None, help='explicit groups for val')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--move', action='store_true', help='move files instead of copying')
    ap.add_argument('--names', default='crop,weed', help='class names for the output data.yaml')
    ap.add_argument('--verify-only', action='store_true', help='only check --out for leakage')
    args = ap.parse_args()

    pattern = re.compile(args.group_pattern)
    out = Path(args.out)
    if args.verify_only:
        verify(out, pattern)
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

    # ---- group frames ------------------------------------------------------
    by_group = defaultdict(list)
    for p in imgs:
        by_group[group_of(video_id_of(p), pattern)].append(p)
    groups = sorted(by_group)
    if len(groups) < 2:
        raise SystemExit(f'Only {len(groups)} group(s) found — a group-level split needs at least 2. '
                         f'Collect more videos/fields; do NOT fall back to a random frame split.')

    # ---- choose val groups -------------------------------------------------
    if args.val_groups:
        unknown = set(args.val_groups) - set(groups)
        if unknown:
            raise SystemExit(f'--val-groups not found: {sorted(unknown)}. Available: {groups}')
        val_groups = set(args.val_groups)
    else:
        order = groups[:]
        random.Random(args.seed).shuffle(order)
        target = args.val_frac * len(imgs)
        val_groups, n_val = set(), 0
        for g in order:
            if n_val >= target:
                break
            val_groups.add(g)
            n_val += len(by_group[g])
    if val_groups == set(groups):
        val_groups.discard(sorted(val_groups)[0])
    if not val_groups:
        raise SystemExit('Val set would be empty.')

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
            manifest.append((p.name, video_id_of(p), g, split))

    with open(out / 'split_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'video_id', 'group', 'split'])
        w.writerows(manifest)

    names = [n.strip() for n in args.names.split(',')]
    (out / 'data.yaml').write_text(
        f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n'
        f'nc: {len(names)}\nnames: {names}\n', encoding='utf-8')

    # ---- hard leakage check on what is actually on disk --------------------
    verify(out, pattern)
    print(f'val groups: {sorted(val_groups)}')
    if n_bg:
        print(f'Note: {n_bg} image(s) had no label file and were kept as background images.')
    print(f'Manifest: {out / "split_manifest.csv"}\nData yaml: {out / "data.yaml"}')


if __name__ == '__main__':
    main()
