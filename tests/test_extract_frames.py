import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import extract_frames


def write_video(path: Path, n_frames=20, fps=10, size=(64, 48)) -> bool:
    """Tiny synthetic mp4. Returns False if this OpenCV build cannot encode it."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        return False
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 12) % 255, np.uint8))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def run_extract(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['extract_frames.py', *map(str, argv)])
    extract_frames.main()


def test_invalid_video_name_fails_before_any_frame_is_written(tmp_path, monkeypatch):
    videos = tmp_path / 'videos'
    videos.mkdir()
    # the valid video sorts FIRST, so a late check would already have extracted it
    good = videos / 'beet_F01_20270415_cam3.mp4'
    if not write_video(good):
        good.write_bytes(b'not decoded: validation fails before any file is opened')
    (videos / 'sunflower_F01_20270415_cam3.mp4').write_bytes(b'placeholder')
    (videos / 'noname.mp4').write_bytes(b'placeholder')
    out = tmp_path / 'frames'

    with pytest.raises(SystemExit) as e:
        run_extract(monkeypatch, videos, '--out', out)

    msg = str(e.value.code)
    assert 'nothing was extracted' in msg
    assert 'sunflower_F01_20270415_cam3.mp4' in msg and 'Unknown crop' in msg
    assert 'noname.mp4' in msg and 'Malformed' in msg
    reported = [ln.strip() for ln in msg.splitlines() if ln.startswith('  ')]   # one line per bad file
    assert len(reported) == 2, reported
    assert not any(ln.startswith('beet_F01_20270415_cam3.mp4') for ln in reported), \
        'a valid name must not be reported'
    assert not out.exists(), 'no output folder and no frames may exist after a validation error'


def test_validate_video_names_returns_ids_for_valid_names(tmp_path):
    vids = [tmp_path / 'beet_F01_20270415_cam3.mp4', tmp_path / 'corn_F02_20270502_cam1.mov']
    assert extract_frames.validate_video_names(vids) == ['beet_F01_20270415_cam3',
                                                         'corn_F02_20270502_cam1']


def test_valid_name_extracts_at_2_fps(tmp_path, monkeypatch):
    video = tmp_path / 'beet_F01_20270415_cam3.mp4'
    if not write_video(video, n_frames=20, fps=10):          # 2 s of 10 fps video
        pytest.skip('this OpenCV build cannot write mp4 test videos')
    out = tmp_path / 'frames'
    run_extract(monkeypatch, video, '--out', out)
    names = sorted(p.name for p in out.glob('*.jpg'))
    # 2 FPS from a 10 fps video: one frame every 5 native frames
    assert names == [f'beet_F01_20270415_cam3__f{i:06d}.jpg' for i in (0, 5, 10, 15)]
