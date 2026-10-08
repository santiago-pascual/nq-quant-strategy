from src.paper.systemd_notify import notify_systemd


def test_systemd_notify_is_a_noop_when_not_running_under_systemd(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert notify_systemd("READY=1") is False
