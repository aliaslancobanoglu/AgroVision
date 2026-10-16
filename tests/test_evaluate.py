import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import evaluate as ev

W = 640                                    # image width; 5 nozzle strips of 128 px
NAMES = {0: 'wheat', 1: 'beet', 2: 'corn', 3: 'weed'}
WHEAT, BEET, CORN, WEED = 0, 1, 2, 3


def box(strip, cls):
    """A YOLO label line: a 60 px box well inside nozzle strip `strip`."""
    return f'{cls} {(64 + 128 * strip) / W} 0.5 {60 / W} {60 / W}'


def det_xyxy(strip):
    cx = 64 + 128 * strip
    return [cx - 30, 290, cx + 30, 350]


def make_image(labels_dir: Path, stem, gt, dets):
    """One synthetic evaluation record. gt = [(strip, class)], dets = [(strip, class, conf)].
    Only the label file is written; inference results are built directly (no model needed)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f'{stem}.txt').write_text('\n'.join(box(s, c) for s, c in gt) + '\n')
    return {
        'path': Path(f'{stem}.jpg'), 'w': W, 'h': W,
        'cls': np.array([c for _, c, _ in dets], dtype=int),
        'xyxy': np.array([det_xyxy(s) for s, _, _ in dets], dtype=float).reshape(-1, 4),
        'conf': np.array([cf for _, _, cf in dets], dtype=float),
    }


def evaluate_at_lowest_conf(per_image, labels_dir, crop_mode):
    """Run prepare() + sweep() per crop and return {crop: row at conf 0.05 + n_gt + oracle}."""
    ev.prepare(per_image, labels_dir, NAMES, crop_mode)
    result = {}
    for crop in sorted({d['crop'] for d in per_image}):
        subset = [d for d in per_image if d['crop'] == crop]
        df, n_gt, _, oracle = ev.sweep(subset)
        result[crop] = (df[df['conf'] == 0.05].iloc[0], n_gt, oracle)
    return result


def test_volunteer_wheat_in_a_beet_field_counts_as_a_weed(tmp_path):
    labels = tmp_path / 'labels'
    per_image = [
        # beet field: a weed (strip 0) and a volunteer WHEAT plant (strip 2); the beet (strip 4) is the crop
        make_image(labels, 'beet_F01_20270415_cam1__f000001',
                   gt=[(0, WEED), (2, WHEAT), (4, BEET)],
                   dets=[(0, WEED, .9), (2, WEED, .8), (4, BEET, .9)]),
        # the detector calls the volunteer wheat "wheat": still a spray class in a beet field -> found
        make_image(labels, 'beet_F01_20270415_cam1__f000002',
                   gt=[(1, WHEAT)], dets=[(1, WHEAT, .7)]),
    ]
    row, n_gt, oracle = evaluate_at_lowest_conf(per_image, labels, 'auto')['beet']
    assert n_gt == 3                                   # weed + 2 volunteer wheat; the beet is not a target
    assert row['strip_recall'] == 1.0 and row['weed_recall'] == 1.0
    assert row['treated_weeds'] == 3
    assert row['spray_rate'] == pytest.approx(3 / 10) == oracle   # 3 strips sprayed out of 2 x 5


def test_the_fields_own_crop_is_never_sprayed(tmp_path):
    labels = tmp_path / 'labels'
    per_image = [
        # corn field: wheat is a weed here, corn is the crop; the detector only sees the corn plant
        make_image(labels, 'corn_F02_20270502_cam1__f000001',
                   gt=[(1, WHEAT), (3, CORN)], dets=[(3, CORN, .9)]),
        # wheat field: the same wheat class is now the crop -> not a target, not sprayed
        make_image(labels, 'wheat_F01_20270301_cam1__f000001',
                   gt=[(1, WHEAT), (3, WEED)], dets=[(1, WHEAT, .9), (3, WEED, .9)]),
    ]
    res = evaluate_at_lowest_conf(per_image, labels, 'auto')

    corn, n_gt, oracle = res['corn']
    assert n_gt == 1                                   # only the wheat weed; corn itself is not a target
    assert corn['spray_rate'] == 0.0                   # the detected corn plant never fires a nozzle
    assert corn['strip_recall'] == 0.0                 # the missed wheat weed is untreated
    assert oracle == pytest.approx(1 / 5)

    wheat, n_gt, oracle = res['wheat']
    assert n_gt == 1                                   # only the weed; the wheat crop is not a target
    assert wheat['sprayed_strips'] == 1                # the strip with the weed only
    assert wheat['spray_rate'] == pytest.approx(1 / 5)
    assert wheat['strip_recall'] == 1.0


def test_a_crop_class_detection_in_its_own_field_adds_no_spraying(tmp_path):
    labels = tmp_path / 'labels'
    # beet field with a weed only; the detector also fires on beet plants in strips 1 and 4
    img = make_image(labels, 'beet_F01_20270415_cam1__f000001',
                     gt=[(0, WEED)],
                     dets=[(0, WEED, .9), (1, BEET, .95), (4, BEET, .95)])
    row, n_gt, _ = evaluate_at_lowest_conf([img], labels, 'auto')['beet']
    assert row['sprayed_strips'] == 1 and row['strip_recall'] == 1.0


def test_fixed_crop_mode_treats_every_image_as_that_crops_field(tmp_path):
    labels = tmp_path / 'labels'
    per_image = [make_image(labels, 'anything__f000001', gt=[(1, WHEAT)], dets=[(1, WHEAT, .9)])]
    res = evaluate_at_lowest_conf(per_image, labels, 'beet')     # file name has no crop; mode supplies it
    row, n_gt, _ = res['beet']
    assert n_gt == 1 and row['strip_recall'] == 1.0              # wheat in a beet field is a weed


def test_auto_mode_rejects_a_file_name_without_a_valid_crop(tmp_path):
    labels = tmp_path / 'labels'
    per_image = [make_image(labels, 'sunflower_F01_20270415_cam1__f000001', gt=[(0, WEED)], dets=[])]
    with pytest.raises(ValueError, match='Unknown crop'):
        ev.prepare(per_image, labels, NAMES, 'auto')


def test_crop_must_be_a_class_in_data_yaml():
    with pytest.raises(ValueError, match='not a class'):
        ev.spray_classes_for('beet', {0: 'wheat', 1: 'corn', 2: 'weed'})


def test_spray_classes_are_every_class_except_the_crop():
    assert ev.spray_classes_for('beet', NAMES) == {WHEAT, CORN, WEED}
    assert ev.spray_classes_for('corn', NAMES) == {WHEAT, BEET, WEED}
    assert ev.spray_classes_for(None, {0: 'crop', 1: 'weed'}) == {1}    # legacy: weed = class 1


# ---- 95% Wilson interval ---------------------------------------------------------------

@pytest.mark.parametrize('k, n, low, high', [
    (50, 100, 0.403832, 0.596168),
    (181, 202, 0.846304, 0.930993),
    (198, 202, 0.950200, 0.992273),
    (3, 7, 0.158220, 0.749542),
])
def test_wilson_interval_known_values(k, n, low, high):
    lo, hi = ev.wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=1e-6) and hi == pytest.approx(high, abs=1e-6)


def test_wilson_interval_edges():
    assert ev.wilson_interval(0, 10)[0] == 0.0
    assert ev.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in ev.wilson_interval(0, 0))


def test_interval_and_warning_are_printed_but_the_verdict_stays_on_the_point_estimate(capsys):
    import pandas as pd
    # 200/202 treated -> point estimate 99.0% reaches the target; Wilson lower bound is below it
    df = pd.DataFrame([{'conf': 0.5, 'strip_recall': 200 / 202, 'weed_recall': 200 / 202,
                        'weed_precision': 0.9, 'spray_rate': 0.4, 'savings': 0.6,
                        'tp': 200, 'fp': 20, 'fn': 2, 'treated_weeds': 200, 'sprayed_strips': 520}])
    passed = ev.verdict(df, oracle=0.3, label='beet', n_gt=202)
    out = capsys.readouterr().out
    assert bool(passed) is True and 'RESULT: PASS' in out
    assert 'Wilson interval' in out and 'GT weeds (n)' in out and '202' in out
    assert 'WARNING' in out and 'below the 98% target' in out
    assert 'optimistic' in out
