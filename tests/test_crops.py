import pytest

from crops import CLASS_NAMES, CROPS, VideoId, parse_video_id, video_id_of_name


def test_valid_name():
    v = parse_video_id('beet_F01_20270415_cam3')
    assert v == VideoId(crop='beet', field='F01', date='20270415', cam='cam3')
    assert v.group == 'beet_F01'


@pytest.mark.parametrize('crop', CROPS)
def test_every_crop_is_accepted(crop):
    assert parse_video_id(f'{crop}_F02_20270501_cam1').crop == crop


def test_unknown_crop():
    with pytest.raises(ValueError, match='Unknown crop'):
        parse_video_id('sunflower_F01_20270415_cam3')


@pytest.mark.parametrize('name', [
    '',                                # empty
    'beet_F01_20270415',               # no camera
    'beet_F01_20270415_3',             # camera is not cam<N>
    'beet_F_01_20270415_cam3',         # field contains an underscore
    'beet-F01-20270415-cam3',          # wrong separators
    'beet_F01_2027415_cam3',           # date is not 8 digits
    'beet_F01_20270415_cam3_extra',    # too many parts
])
def test_malformed_name(name):
    with pytest.raises(ValueError, match='Malformed'):
        parse_video_id(name)


@pytest.mark.parametrize('name', [
    'beet_F01_20271345_cam1',          # month 13
    'beet_F01_20270230_cam1',          # 30 February
])
def test_invalid_date(name):
    with pytest.raises(ValueError, match='Invalid date'):
        parse_video_id(name)


def test_video_id_of_frame_name():
    assert video_id_of_name('beet_F01_20270415_cam3__f000015.jpg') == 'beet_F01_20270415_cam3'
    with pytest.raises(ValueError, match='separator'):
        video_id_of_name('beet_F01_20270415_cam3.jpg')


def test_class_order_is_what_the_docs_say():
    assert CLASS_NAMES == ('wheat', 'beet', 'corn', 'weed')
    assert 'crop' not in CLASS_NAMES
