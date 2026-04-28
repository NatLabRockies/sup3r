"""Tests for training session error propagation."""

import pytest

from sup3r.models.utilities import TrainingSession


class _FakeBatchHandler:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FailingModel:
    def train(self, batch_handler, config=None):
        raise RuntimeError('thread failure')


def test_training_session_reraises_thread_failure():
    """Worker thread failures should propagate to the caller."""

    batch_handler = _FakeBatchHandler()
    session = TrainingSession(
        batch_handler=batch_handler,
        model=_FailingModel(),
        input_resolution={'spatial': '1km', 'temporal': '1h'},
        out_dir='test_{epoch}',
        n_epoch=1,
    )

    with pytest.raises(RuntimeError, match='thread failure'):
        session.run()

    assert batch_handler.stopped
