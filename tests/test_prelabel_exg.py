import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import prelabel_exg
from crops import CLASS_NAMES

SOIL_BGR = (60, 100, 140)           # brown soil
LEAF_BGR = (50, 150, 60)            # green leaf
FALLOW_STEM = 'fallow_F01_20261020_cam1__f{:06d}'


def soil(h=200, w=300, seed=0) -> np.ndarray:
    """Brown soil with mild per-pixel noise, so Otsu has something to split."""
    rng = np.random.default_rng(seed)
    img = np.array(SOIL_BGR, np.int16) + rng.integers(-12, 13, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def write_image(folder: Path, stem: str, img: np.ndarray) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{stem}.png'
    assert cv2.imwrite(str(path), img)
    return path


def run_prelabel(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['prelabel_exg.py', *map(str, argv)])
    prelabel_exg.main()


def test_each_green_blob_on_brown_soil_gives_one_weed_box(tmp_path, monkeypatch, capsys):
    img = soil()
    cv2.circle(img, (60, 50), 15, LEAF_BGR, -1)          # 30 x 30 px blob
    cv2.rectangle(img, (200, 120), (239, 149), LEAF_BGR, -1)   # 40 x 30 px blob
    write_image(tmp_path / 'frames', FALLOW_STEM.format(0), img)
    out = tmp_path / 'prelabels'
    run_prelabel(monkeypatch, '--images', tmp_path / 'frames', '--out', out)

    lines = (out / f'{FALLOW_STEM.format(0)}.txt').read_text().splitlines()
    assert len(lines) == 2
    boxes = sorted([float(v) for v in ln.split()[1:]] for ln in lines)
    assert all(ln.split()[0] == str(CLASS_NAMES.index('weed')) for ln in lines)
    (xc1, yc1, w1, h1), (xc2, yc2, w2, h2) = boxes
    assert (xc1 * 300, yc1 * 200, w1 * 300, h1 * 200) == pytest.approx((60.5, 50.5, 31, 31), abs=2)
    assert (xc2 * 300, yc2 * 200, w2 * 300, h2 * 200) == pytest.approx((220, 135, 40, 30), abs=2)
    assert (out / 'classes.txt').read_text().splitlines() == list(CLASS_NAMES)
    assert 'DRAFTS' in capsys.readouterr().out


def test_bare_soil_gives_no_boxes(tmp_path, monkeypatch):
    for i in range(3):
        write_image(tmp_path / 'frames', FALLOW_STEM.format(i), soil(seed=i))
    out = tmp_path / 'prelabels'
    run_prelabel(monkeypatch, '--images', tmp_path / 'frames', '--out', out)
    for i in range(3):
        assert (out / f'{FALLOW_STEM.format(i)}.txt').read_text() == ''


def test_small_blobs_below_min_area_are_dropped():
    img = soil()
    cv2.circle(img, (60, 50), 15, LEAF_BGR, -1)          # ~700 px
    cv2.circle(img, (200, 120), 4, LEAF_BGR, -1)         # ~50 px
    assert len(prelabel_exg.weed_boxes(img, min_area_px=25)) == 2
    assert len(prelabel_exg.weed_boxes(img, min_area_px=200)) == 1


@pytest.mark.parametrize('stem', ['beet_F01_20270415_cam1__f000000', 'no_video_id'])
def test_a_non_fallow_image_is_refused_and_nothing_is_written(tmp_path, monkeypatch, stem):
    img = soil()
    cv2.circle(img, (60, 50), 15, LEAF_BGR, -1)
    write_image(tmp_path / 'frames', FALLOW_STEM.format(0), img)
    write_image(tmp_path / 'frames', stem, img)
    out = tmp_path / 'prelabels'
    with pytest.raises(SystemExit) as e:
        run_prelabel(monkeypatch, '--images', tmp_path / 'frames', '--out', out)
    assert 'Refusing to pre-label' in str(e.value.code) and stem in str(e.value.code)
    assert not out.exists()


@pytest.mark.parametrize('where', ['labels', 'images'])
def test_drafts_never_land_in_a_labels_or_images_folder(tmp_path, monkeypatch, where):
    frames = tmp_path / 'frames'
    write_image(frames, FALLOW_STEM.format(0), soil())
    out = tmp_path / 'dataset' / 'labels' / 'train' if where == 'labels' else frames
    with pytest.raises(SystemExit):
        run_prelabel(monkeypatch, '--images', frames, '--out', out)
    assert not list(frames.glob('*.txt')) and not (tmp_path / 'dataset').exists()
