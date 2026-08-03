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
