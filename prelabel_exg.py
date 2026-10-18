"""
AgroVision AI — DRAFT pre-labels for FALLOW field frames (Excess Green + Otsu).

In a fallow field (no planted crop: bare soil, stubble) every green plant is a
weed, so "green region" = "weed box" is a usable first draft there. Everywhere
else it is not: ExG cannot tell a crop from a weed. This script therefore
REFUSES any image whose video ID is not a fallow field (crops.parse_video_id),
e.g. it accepts fallow_F01_20261020_cam1__f000003.jpg and refuses beet_... .

Method, per image
-----------------
1. Normalized RGB (chromatic coordinates): r, g, b = R, G, B / (R + G + B).
2. Excess Green: ExG = 2g - r - b.
3. Otsu threshold on ExG. Otsu always splits the histogram in two, so on bare
   soil it would split soil into "lighter" and "darker" soil; the threshold is
   therefore never below MIN_EXG. Very dark pixels (R + G + B < MIN_RGB_SUM)
   have unstable chromatic coordinates and are never vegetation.
4. Morphological opening with a small (3x3) kernel removes isolated pixels.
5. Each connected green region of at least --min-area-px pixels becomes one
   YOLO box with class "weed" (its ID in CLASS_NAMES, crops.py).

Output
------
--out (default prelabels/) receives one <image stem>.txt per image (empty when
no region was found) and a classes.txt in CLASS_NAMES order, so the labeling
tool's export later matches split_dataset.py --export-classes. Nothing is ever
written into a labels folder or next to the images.

THESE ARE DRAFTS. Import them into the labeling tool (Roboflow / CVAT) and have
a human review every frame. The validation set must be 100% human-verified;
otherwise the go/no-go verdict measures the pre-labeler against itself.

Usage
-----
    python prelabel_exg.py --images frames_fallow
    python prelabel_exg.py --images frames_fallow --out prelabels --min-area-px 25
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from crops import CLASS_NAMES, FALLOW, parse_video_id, video_id_of_name

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
WEED_ID = CLASS_NAMES.index('weed')
MIN_EXG = 0.1          # Otsu threshold floor: soil sits near ExG 0, green leaves well above
MIN_RGB_SUM = 30       # pixels darker than this are never vegetation
OPEN_KERNEL = 3        # morphological opening kernel size (px)
DEFAULT_MIN_AREA = 25  # px; a 2 cm weed at ~2 mm/px is ~10 px across


def excess_green(bgr: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on normalized RGB, float32 in [-1, 2]; 0 for very dark pixels."""
    img = bgr.astype(np.float32)
    s = img.sum(axis=2)
    dark = s < MIN_RGB_SUM
    s[s == 0] = 1
    b, g, r = (img[..., i] / s for i in range(3))
    exg = 2 * g - r - b
    exg[dark] = 0
    return exg


def green_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = vegetation): ExG > max(Otsu, MIN_EXG), then a small opening."""
    exg = excess_green(bgr)
    exg8 = np.round((exg + 1) / 3 * 255).astype(np.uint8)
    otsu8, _ = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu8 / 255 * 3 - 1, MIN_EXG)
    mask = (exg > thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def weed_boxes(bgr: np.ndarray, min_area_px: int) -> list:
    """YOLO label lines 'weed_id xc yc w h' (normalized), one per connected green region."""
    h, w = bgr.shape[:2]
    _, _, stats, _ = cv2.connectedComponentsWithStats(green_mask(bgr), connectivity=8)
    lines = []
    for x, y, bw, bh, area in stats[1:]:                 # row 0 is the background
        if area < min_area_px:
            continue
        lines.append(f'{WEED_ID} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} '
                     f'{bw / w:.6f} {bh / h:.6f}')
    return lines


def check_fallow(images) -> None:
    """Hard error, listing every offender, unless ALL images come from fallow fields."""
    bad = []
    for p in images:
        try:
            crop = parse_video_id(video_id_of_name(p.name)).crop
        except ValueError as e:
            bad.append(f'  {p.name}: {e}')
            continue
        if crop != FALLOW:
            bad.append(f"  {p.name}: field type '{crop}', not '{FALLOW}'")
    if bad:
        raise SystemExit('Refusing to pre-label: ExG cannot tell a crop from a weed, so only '
                         f'{FALLOW} field frames (e.g. {FALLOW}_F01_20261020_cam1__f000000.jpg) '
                         'are allowed. Nothing was written.\n' + '\n'.join(bad))


def check_out_dir(out: Path, img_dir: Path) -> None:
    if 'labels' in (part.lower() for part in out.resolve().parts):
        raise SystemExit(f'--out {out} is inside a labels folder. Pre-labels are drafts and must '
                         f'never land where real labels live; use e.g. prelabels/.')
    if out.resolve() == img_dir.resolve():
        raise SystemExit(f'--out {out} is the images folder: draft .txt files next to the frames '
                         f'would be picked up as real labels by split_dataset.py.')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty — refusing to overwrite earlier drafts. '
                         f'Choose a new --out.')


def main():
    ap = argparse.ArgumentParser(description='Draft ExG weed boxes for fallow field frames.')
    ap.add_argument('--images', required=True, help='flat folder of fallow frames')
    ap.add_argument('--out', default='prelabels', help='output folder for the drafts (must be new/empty)')
    ap.add_argument('--min-area-px', type=int, default=DEFAULT_MIN_AREA,
                    help=f'smallest green region kept, in pixels (default {DEFAULT_MIN_AREA})')
    args = ap.parse_args()

    img_dir, out = Path(args.images), Path(args.out)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS) \
        if img_dir.is_dir() else []
    if not images:
        raise SystemExit(f'No images in {img_dir}')
    check_fallow(images)
    check_out_dir(out, img_dir)

    out.mkdir(parents=True, exist_ok=True)
    (out / 'classes.txt').write_text('\n'.join(CLASS_NAMES) + '\n', encoding='utf-8')
    n_boxes = n_empty = 0
    for p in images:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f'Cannot read image {p}')
        lines = weed_boxes(bgr, args.min_area_px)
        (out / (p.stem + '.txt')).write_text(''.join(ln + '\n' for ln in lines), encoding='utf-8')
        n_boxes += len(lines)
        n_empty += not lines

    print(f'{len(images)} image(s), {n_boxes} draft weed box(es) '
          f'({n_empty} image(s) without any) -> {out}')
    print(f'classes.txt ({", ".join(CLASS_NAMES)}) -> {out / "classes.txt"}')
    print('REMINDER: these are DRAFTS. Review and correct every frame in the labeling tool '
          '(Roboflow / CVAT). The validation set must be 100% human-verified.')


if __name__ == '__main__':
    main()
