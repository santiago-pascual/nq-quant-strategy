import pytest

from src.paper import atomic_io


def test_native_premove_failure_retries(monkeypatch, tmp_path):
    calls = []
    def replace(source, target):
        calls.append(1)
        if len(calls) == 1:
            error = OSError('pre-move failure')
            error.winerror = 1175
            raise error
    monkeypatch.setattr(atomic_io, '_replace_once', replace)
    atomic_io.atomic_replace_with_retry(tmp_path/'source', tmp_path/'target',
                                        attempts=2, initial_delay_seconds=0)
    assert len(calls) == 2


@pytest.mark.parametrize('code', [1176, 1177])
def test_native_partial_failure_is_not_retried(monkeypatch, tmp_path, code):
    calls = []
    def replace(source, target):
        calls.append(1)
        error = OSError('native replacement cannot be safely retried')
        error.winerror = code
        raise error
    monkeypatch.setattr(atomic_io, '_replace_once', replace)
    with pytest.raises(OSError):
        atomic_io.atomic_replace_with_retry(tmp_path/'source', tmp_path/'target',
                                            attempts=5, initial_delay_seconds=0)
    assert len(calls) == 1
