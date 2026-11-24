import json
from types import SimpleNamespace
import pytest
from anveshak.train import pause_after_save


def T(tmp_path, epoch, epochs):
    return SimpleNamespace(save_dir=str(tmp_path), epoch=epoch, epochs=epochs)


def test_no_flag_keeps_training(tmp_path):
    pause_after_save(T(tmp_path, 0, 10))  # returns normally


def test_flag_exits_after_save_and_marks_paused(tmp_path):
    (tmp_path / "PAUSE").touch()
    with pytest.raises(SystemExit):
        pause_after_save(T(tmp_path, 3, 10))
    assert json.loads((tmp_path / "state.json").read_text())["state"] == "paused"
    assert not (tmp_path / "PAUSE").exists()  # so resume doesn't pause again immediately


def test_final_epoch_ignores_flag(tmp_path):
    (tmp_path / "PAUSE").touch()
    pause_after_save(T(tmp_path, 9, 10))  # last epoch: finish normally (final eval runs)


def test_stop_flag_raises_keyboardinterrupt_once(tmp_path):
    from anveshak.train import stop_now
    stop_now(T(tmp_path, 0, 10))  # no flag: nothing happens
    (tmp_path / "STOP").touch()
    with pytest.raises(KeyboardInterrupt):
        stop_now(T(tmp_path, 0, 10))
    assert not (tmp_path / "STOP").exists()
