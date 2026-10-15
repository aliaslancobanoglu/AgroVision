"""
AgroVision AI — extract frames from field video at a fixed TARGET FPS (default 2).

Why 2 FPS and not the video's native framerate:
consecutive frames at 30-60 fps are near-duplicates (the boom moves ~7-14 cm
between frames at 15 km/h). They multiply annotation cost without adding new
information, and they make train/val leakage far more damaging. Sampling is
time-based, so the result is the same regardless of the camera's native fps.

Output naming (split_dataset.py depends on this — do not change casually):

    <video_id>__f<frame_index:06d>.jpg

video_id is the source video's filename stem. Name videos so the field is
recoverable from it, e.g.  FIELD01_20270415_cam3.mp4 -> video_id
"FIELD01_20270415_cam3", field "FIELD01". Avoid "__" inside video filenames.

Usage
-----
    python extract_frames.py videos/                    # every video in a folder
    python extract_frames.py a.mp4 b.mp4 --fps 2 --out frames
"""

import argparse
import re
from pathlib import Path

import cv2

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mts', '.ts'}
SEP = '__'
DEFAULT_FPS = 2.0


def video_id_for(path: Path) -> str:
    vid = re.sub(r'[^A-Za-z0-9_-]', '-', path.stem)
    if SEP in vid:
        raise ValueError(f"Video name '{path.name}' contains '{SEP}', which is reserved "
                         f"as the video-id separator. Rename the file.")
    return vid


def extract(path: Path, out_dir: Path, target_fps: float, jpeg_quality: int) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native <= 0:
        raise RuntimeError(f'{path}: native fps unknown; cannot sample by time.')
    if target_fps >= native:
        print(f'  WARNING: target {target_fps} fps >= native {native:.2f} fps; keeping every frame.')

    vid = video_id_for(path)
    step = 1.0 / target_fps
    next_t = 0.0
    idx = saved = 0
    while cap.grab():                      # grab() skips decoding of unused frames
        t = idx / native
        if t + 1e-9 >= next_t:
            ok, frame = cap.retrieve()
            if ok:
                name = f'{vid}{SEP}f{idx:06d}.jpg'
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                saved += 1
            next_t += step
            while next_t <= t:             # guard against drift
                next_t += step
        idx += 1
    cap.release()
    print(f'  {path.name}: {idx} frames @ {native:.2f} fps ({idx / native:.1f} s) '
          f'-> {saved} extracted @ {target_fps} fps')
    return saved


def main():
    ap = argparse.ArgumentParser(description='Extract frames from video at a target FPS.')
    ap.add_argument('inputs', nargs='+', help='video files and/or folders of videos')
    ap.add_argument('--fps', type=float, default=DEFAULT_FPS,
                    help=f'target frames per second (default {DEFAULT_FPS}); NOT the native fps')
    ap.add_argument('--out', default='frames')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality')
    args = ap.parse_args()
    if args.fps <= 0:
        raise SystemExit('--fps must be > 0')

    videos = []
    for s in map(Path, args.inputs):
        if s.is_dir():
            videos += sorted(p for p in s.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        elif s.is_file():
            videos.append(s)
        else:
            raise FileNotFoundError(s)
    if not videos:
        raise SystemExit('No videos found.')

    ids = [video_id_for(v) for v in videos]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f'Duplicate video IDs (same filename stem): {sorted(dupes)}. Rename first.')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {len(videos)} video(s) at {args.fps} fps -> {out_dir}')
    total = 0
    for v in videos:
        total += extract(v, out_dir, args.fps, args.quality)
    print(f'\nTotal frames extracted: {total} from {len(videos)} video(s)')


if __name__ == '__main__':
    main()
