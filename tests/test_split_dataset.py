import csv
import sys
from pathlib import Path

import pytest

import split_dataset
from crops import CLASS_NAMES

GOOD_VIDEOS = [
    'beet_F01_20270415_cam1', 'beet_F02_20270416_cam1', 'beet_F03_20270417_cam1',
    'corn_F01_20270501_cam1', 'corn_F02_20270502_cam1',
    'wheat_F01_20270301_cam1', 'wheat_F02_20270302_cam1',
]
ALPHABETICAL = ['beet', 'corn', 'weed', 'wheat']        # what a labeling tool may export


def make_frames(folder: Path, videos, class_ids=(0,), frames_per_video=3) -> Path:
    """Placeholder frames (split_dataset never decodes images) plus one label file each,
    holding one box per class id in `class_ids`."""
    folder.mkdir(parents=True, exist_ok=True)
    for v in videos:
        for i in range(frames_per_video):
            stem = f'{v}__f{i:06d}'
            (folder / f'{stem}.jpg').write_bytes(b'placeholder')
            (folder / f'{stem}.txt').write_text(
                ''.join(f'{c} 0.5 0.5 0.1 0.1\n' for c in class_ids))
    return folder


def write_classes(path: Path, names) -> Path:
    path.write_text('\n'.join(names) + '\n', encoding='utf-8')
    return path


def run_split(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['split_dataset.py', *map(str, argv)])
    split_dataset.main()


def exit_message(excinfo) -> str:
    return str(excinfo.value.code)


def test_good_split(tmp_path, monkeypatch, capsys):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    classes = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', classes)

    with open(out / 'split_manifest.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(GOOD_VIDEOS) * 3
    groups = {}
    for r in rows:
        groups.setdefault(r['group'], set()).add(r['split'])
    assert all(len(s) == 1 for s in groups.values()), 'a field was split across train and val'
    for crop in ('beet', 'corn', 'wheat'):
        splits = {next(iter(s)) for g, s in groups.items() if g.startswith(crop + '_')}
        assert splits == {'train', 'val'}, f'{crop} needs a field in train AND in val'
    split_dataset.verify(out)                             # the on-disk leakage check passes too
    assert "names: ['wheat', 'beet', 'corn', 'weed']" in (out / 'data.yaml').read_text()
    # classes already in CLASS_NAMES order -> nothing changes
    assert 'CHANGED' not in capsys.readouterr().out


def test_crop_with_a_single_field_is_a_hard_error(tmp_path, monkeypatch):
    videos = [v for v in GOOD_VIDEOS if not v.startswith('wheat_F02')]
    frames = make_frames(tmp_path / 'frames', videos)
    classes = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', classes)
    assert "Crop 'wheat' has only 1 field" in exit_message(e)
    assert not (out / 'images').exists()


def test_pinned_split_that_empties_a_crop_is_a_hard_error(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    classes = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    # both corn fields pinned to val -> corn has nothing left in train
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', classes,
                  '--val-groups', 'corn_F01', 'corn_F02', 'beet_F01', 'wheat_F01')
    assert "crop 'corn'" in exit_message(e)
    assert not (out / 'images').exists()


def read_labels(out: Path):
    files = sorted((out / 'labels').rglob('*.txt'))
    assert files, 'no label files were written'
    return {f.name: f.read_text().splitlines() for f in files}


@pytest.mark.parametrize('kind', ['classes.txt', 'yaml-list', 'yaml-dict'])
def test_class_ids_are_remapped_by_name_from_an_alphabetical_export(tmp_path, monkeypatch, capsys, kind):
    # every label holds one box per ALPHABETICAL class, i.e. export IDs 0,1,2,3 = beet,corn,weed,wheat
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS, class_ids=(0, 1, 2, 3))
    if kind == 'classes.txt':
        export = write_classes(tmp_path / 'classes.txt', ALPHABETICAL)
    elif kind == 'yaml-list':
        export = tmp_path / 'data.yaml'
        export.write_text(f'nc: 4\nnames: {ALPHABETICAL}\n')
    else:
        export = tmp_path / 'data.yaml'
        export.write_text('nc: 4\nnames:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(ALPHABETICAL)))
    out = tmp_path / 'out'
    run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)

    expected = [CLASS_NAMES.index(n) for n in ALPHABETICAL]        # beet->1, corn->2, weed->3, wheat->0
    assert expected == [1, 2, 3, 0]
    for name, lines in read_labels(out).items():
        assert [int(ln.split()[0]) for ln in lines] == expected, name
        assert all(ln.split()[1:] == ['0.5', '0.5', '0.1', '0.1'] for ln in lines), 'box was altered'
    printed = capsys.readouterr().out
    assert '0 beet -> 1 beet' in printed and '3 wheat -> 0 wheat' in printed and 'CHANGED' in printed
    assert "names: ['wheat', 'beet', 'corn', 'weed']" in (out / 'data.yaml').read_text()


def test_a_single_class_follows_its_name_not_its_number(tmp_path, monkeypatch):
    # a beet-only label in an alphabetical export is ID 0 ('beet'); in the dataset beet is ID 1
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS, class_ids=(0,))
    export = write_classes(tmp_path / 'classes.txt', ALPHABETICAL)
    out = tmp_path / 'out'
    run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    assert all(lines == ['1 0.5 0.5 0.1 0.1'] for lines in read_labels(out).values())


def test_unknown_and_missing_class_names_are_hard_errors(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    out = tmp_path / 'out'
    export = write_classes(tmp_path / 'classes.txt', ['wheat', 'beet', 'corn', 'sunflower'])
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    msg = exit_message(e)
    assert 'unknown class name' in msg and 'sunflower' in msg
    assert 'missing from the export' in msg and 'weed' in msg
    assert not (out / 'images').exists()


def test_export_missing_a_class_is_a_hard_error(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    out = tmp_path / 'out'
    export = write_classes(tmp_path / 'classes.txt', ['beet', 'weed', 'wheat'])
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    assert 'missing from the export' in exit_message(e) and 'corn' in exit_message(e)
    assert not (out / 'images').exists()


@pytest.mark.parametrize('bad_id', ['7', '4', '-1'])
def test_out_of_range_class_id_is_a_hard_error_and_nothing_is_written(tmp_path, monkeypatch, bad_id):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS, class_ids=(0,))
    # corrupt one label file only; the error must still stop the whole run before any output
    victim = sorted(frames.glob('*.txt'))[-1]
    victim.write_text(f'{bad_id} 0.5 0.5 0.1 0.1\n')
    export = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    assert 'out of range' in exit_message(e) and victim.name in exit_message(e)
    assert not out.exists() or not any(out.rglob('*.jpg'))


def test_export_classes_is_required(tmp_path, monkeypatch, capsys):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', tmp_path / 'out')
    assert e.value.code == 2
    assert '--export-classes' in capsys.readouterr().err


def test_verify_only_does_not_need_export_classes(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS)
    export = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    run_split(monkeypatch, '--verify-only', '--out', out)      # must not raise


def test_unknown_crop_in_a_video_name_is_a_hard_error(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS + ['sunflower_F01_20270415_cam1'])
    export = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='Unknown crop'):
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', export)
    assert not (out / 'images').exists()


def test_fallow_fields_are_split_like_any_crop(tmp_path, monkeypatch):
    videos = GOOD_VIDEOS + ['fallow_F01_20261020_cam1', 'fallow_F02_20261021_cam1']
    frames = make_frames(tmp_path / 'frames', videos)
    classes = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', classes)
    with open(out / 'split_manifest.csv', newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['crop'] == 'fallow']
    assert {r['group']: r['split'] for r in rows} in (
        {'fallow_F01': 'train', 'fallow_F02': 'val'}, {'fallow_F01': 'val', 'fallow_F02': 'train'})
    # a field type, not a class
    assert "names: ['wheat', 'beet', 'corn', 'weed']" in (out / 'data.yaml').read_text()


def test_a_single_fallow_field_is_a_hard_error(tmp_path, monkeypatch):
    frames = make_frames(tmp_path / 'frames', GOOD_VIDEOS + ['fallow_F01_20261020_cam1'])
    classes = write_classes(tmp_path / 'classes.txt', CLASS_NAMES)
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run_split(monkeypatch, '--images', frames, '--out', out, '--export-classes', classes)
    assert "Crop 'fallow' has only 1 field" in exit_message(e)
    assert not (out / 'images').exists()
