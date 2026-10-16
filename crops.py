"""
AgroVision AI — crop / video-ID naming convention shared by split_dataset.py and evaluate.py.

Video naming convention:

    <crop>_<field>_<YYYYMMDD>_<cam>        e.g.  beet_F01_20270415_cam3

  crop   one of CROPS (wheat, beet, corn)
  field  alphanumeric field ID (no underscore)
  date   recording date, YYYYMMDD
  cam    cam<N>

Extracted frames are named <video_id>__f<index>.<ext> (extract_frames.py), so
the video ID is everything before the first '__'.

Multi-crop class order (labels describe what the plant IS, never "crop"):

    0 = wheat, 1 = beet, 2 = corn, 3 = weed
"""

import re
from datetime import datetime
from typing import NamedTuple

CROPS = ('wheat', 'beet', 'corn')
CLASS_NAMES = ('wheat', 'beet', 'corn', 'weed')
SEP = '__'

_VIDEO_RE = re.compile(r'^([^_]+)_([A-Za-z0-9]+)_(\d{8})_(cam\d+)$')


class VideoId(NamedTuple):
    crop: str
    field: str
    date: str
    cam: str

    @property
    def group(self) -> str:
        """Split unit: whole crop_field groups go to train or val."""
        return f'{self.crop}_{self.field}'


def parse_video_id(video_id: str) -> VideoId:
    """Parse '<crop>_<field>_<YYYYMMDD>_<cam>'. Hard error (ValueError) on anything else."""
    m = _VIDEO_RE.match(video_id)
    if not m:
        raise ValueError(f"Malformed video ID '{video_id}'. Expected "
                         f"<crop>_<field>_<YYYYMMDD>_<cam>, e.g. beet_F01_20270415_cam3.")
    crop, field, date, cam = m.groups()
    if crop not in CROPS:
        raise ValueError(f"Unknown crop '{crop}' in video ID '{video_id}'. "
                         f"Crop must be one of {', '.join(CROPS)}.")
    try:
        datetime.strptime(date, '%Y%m%d')
    except ValueError:
        raise ValueError(f"Invalid date '{date}' in video ID '{video_id}' (expected YYYYMMDD).")
    return VideoId(crop, field, date, cam)


def video_id_of_name(name: str) -> str:
    """Video ID of a frame file name or stem: everything before the first '__'."""
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    if SEP not in stem:
        raise ValueError(f"'{name}' has no '{SEP}' separator — cannot tell which video it "
                         f"came from. Frames must be named by extract_frames.py.")
    return stem.split(SEP)[0]
