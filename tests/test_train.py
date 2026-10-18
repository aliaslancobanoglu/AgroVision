import pytest

import train


class FakeYOLO:
    """Stands in for ultralytics.YOLO: records the start weights and train() arguments."""
    calls = []

    def __init__(self, weights):
        self.weights = weights

    def train(self, **kwargs):
        FakeYOLO.calls.append((self.weights, kwargs))


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """train.main() with a temp project root, a fake GPU and no real training."""
    FakeYOLO.calls = []
    monkeypatch.chdir(tmp_path)               # main() chdirs to ROOT; restore cwd afterwards
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(train, 'YOLO', FakeYOLO)
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train.torch.cuda, 'get_device_name', lambda i: 'Fake GPU')
    return tmp_path


def test_defaults_equal_the_previous_hardcoded_settings():
    a = train.parse_args([])
    assert (a.data, a.weights, a.run_name) == ('data.yaml', 'yolo11n.pt', 'kaggle_yolo11n_640')
    assert (a.imgsz, a.batch, a.epochs, a.patience) == (640, 16, 100, 20)


def test_a_bare_run_trains_with_the_defaults(fake_env, capsys):
    train.main([])
    weights, kw = FakeYOLO.calls[0]
    assert weights == 'yolo11n.pt'
    assert {k: kw[k] for k in ('data', 'imgsz', 'batch', 'epochs', 'patience', 'name')} == {
        'data': 'data.yaml', 'imgsz': 640, 'batch': 16, 'epochs': 100, 'patience': 20,
        'name': 'kaggle_yolo11n_640'}
    assert kw['exist_ok'] is False
    out = capsys.readouterr().out
    assert 'yolo11n.pt' in out and 'imgsz 640 | batch 16 | epochs 100 | patience 20' in out


def test_options_override_the_defaults(fake_env, capsys):
    train.main(['--data', 'dataset_konya/data.yaml', '--weights', 'runs/a/weights/best.pt',
                '--run-name', 'konya_1280', '--imgsz', '1280', '--batch', '4',
                '--epochs', '50', '--patience', '10'])
    weights, kw = FakeYOLO.calls[0]
    assert weights == 'runs/a/weights/best.pt'
    assert (kw['data'], kw['imgsz'], kw['batch'], kw['epochs'], kw['patience'], kw['name']) == (
        'dataset_konya/data.yaml', 1280, 4, 50, 10, 'konya_1280')
    assert 'konya_1280' in capsys.readouterr().out


def test_an_existing_run_folder_is_never_overwritten(fake_env):
    (fake_env / 'runs' / 'konya_1280').mkdir(parents=True)
    with pytest.raises(SystemExit, match='already exists'):
        train.main(['--run-name', 'konya_1280'])
    assert FakeYOLO.calls == []


def test_training_on_cpu_is_refused(fake_env, monkeypatch):
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(SystemExit, match='Refusing to train on CPU'):
        train.main([])
    assert FakeYOLO.calls == []
