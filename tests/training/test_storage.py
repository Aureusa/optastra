import pytest

from optastra.training.storage import EventStorage


def test_event_storage_tracks_latest_history_and_smoothed_values():
    storage = EventStorage(start_iter=0, window_size=3)

    storage.put_scalar("loss", 3.0)
    storage.iter = 1
    storage.put_scalar("loss", 2.0)
    storage.iter = 2
    storage.put_scalar("loss", 1.0)

    assert storage.latest()["loss"] == 1.0
    assert storage.history("loss") == [(0, 3.0), (1, 2.0), (2, 1.0)]
    assert storage.smoothed("loss") == 2.0


def test_event_storage_keeps_eval_axis_separate_from_train_axis():
    storage = EventStorage(window_size=5)
    storage.iter = 3
    storage.put_scalar("total_loss", 1.0)

    storage.eval_iter = 3
    storage.put_scalar("val_step_total_loss", 9.0, axis="eval_iter")
    storage.put_scalar("total_loss", 100.0, axis="eval_iter")  # same name, other namespace

    # Train-axis readers never see eval-step scalars (and vice versa).
    assert storage.keys() == ["total_loss"]
    assert storage.smoothed("total_loss") == 1.0
    assert storage.latest_fresh(max_age=0) == {"total_loss": 1.0}
    assert storage.latest_fresh(max_age=0, axis="eval_iter") == {"val_step_total_loss": 9.0, "total_loss": 100.0}

    storage.reset_eval()
    assert storage.keys(axis="eval_iter") == []
    assert storage.eval_iter == 0
    assert storage.latest() == {"total_loss": 1.0}


def test_event_storage_reading_unknown_name_does_not_create_it():
    storage = EventStorage()
    assert storage.smoothed("missing") != storage.smoothed("missing")  # nan
    assert storage.history("missing") == []
    assert storage.keys() == []


def test_event_storage_rejects_unknown_axis():
    with pytest.raises(ValueError, match="axis"):
        EventStorage().put_scalar("x", 1.0, axis="epoch")
