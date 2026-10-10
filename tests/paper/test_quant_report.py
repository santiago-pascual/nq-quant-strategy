from datetime import date
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace

import pytest

from src.paper.quant_report import period_bounds, closed_trade_drawdown


def test_daily_report_handles_23_hour_dst_day():
    start, end = period_bounds(date(2026, 3, 8), 'daily')
    assert start == '2026-03-08T05:00:00+00:00'
    assert end == '2026-03-09T04:00:00+00:00'


def test_daily_report_handles_25_hour_dst_day():
    start, end = period_bounds(date(2026, 11, 1), 'daily')
    assert start == '2026-11-01T04:00:00+00:00'
    assert end == '2026-11-02T05:00:00+00:00'


def test_weekly_report_uses_monday_and_exclusive_next_monday():
    assert period_bounds(date(2026, 10, 10), 'weekly') == (
        '2026-10-05T04:00:00+00:00', '2026-10-12T04:00:00+00:00')


def test_invalid_report_period_is_rejected():
    with pytest.raises(ValueError):
        period_bounds(date(2026, 10, 10), 'annual')


def test_first_loss_is_included_in_sample_drawdown():
    result = closed_trade_drawdown([{'net_pnl': -239.22}])
    assert result['current_usd'] == -239.22
    assert result['maximum_usd'] == -239.22
    assert closed_trade_drawdown([])['available'] is False


def test_closed_trade_sample_drawdown_uses_zero_baseline_and_peak():
    result = closed_trade_drawdown([{'net_pnl': -100}, {'net_pnl': 200}, {'net_pnl': -150}])
    assert result['maximum_usd'] == -150
    assert result['current_usd'] == -150
    assert closed_trade_drawdown([{'net_pnl': None}])['available'] is False


def test_report_delivery_deduplicates_across_invocations(monkeypatch):
    import src.paper.quant_report as module
    import src.paper.notification_credentials as credentials
    import src.paper.notifications as notifications
    import src.paper.notification_cli as notification_cli
    scratch = Path(__file__).resolve().parents[2] / '.test_scratch'
    scratch.mkdir(exist_ok=True)
    sent = []
    monkeypatch.setattr(credentials, 'load_credentials', lambda: SimpleNamespace(token='fixture', chat_id='fixture'))
    monkeypatch.setattr(notifications, 'TelegramTransport', lambda *args: sent.append)
    monkeypatch.setattr(notification_cli, '_rate_limiter', lambda run: None)
    monkeypatch.setattr(module, 'build_report', lambda *args, **kwargs: {
        'report_id': 'test-report-stable-id', 'period_start_utc': '2026-10-08T04:00:00Z',
        'period_end_utc_exclusive': '2026-10-09T04:00:00Z', 'period_has_ended': True,
        'metrics': {'trades': 1, 'net_pnl': -239.22}})
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        run = Path(temp) / 'run'; run.mkdir()
        args = ['--run-dir', str(run), '--date', '2026-10-08', '--send',
                '--output-dir', str(Path(temp) / 'reports')]
        assert module.main(args) == 0
        assert module.main(args) == 0
        assert len(sent) == 1
        assert 'Internal simulated fills only' in sent[0]
        assert list(run.iterdir()) == []


def test_existing_notifier_retries_report_spool_after_restart(monkeypatch):
    import src.paper.notification_cli as module
    from src.paper.notifications import PersistentEventNotifier, TelegramDeliveryError
    root = Path(__file__).resolve().parents[2] / '.test_scratch'
    root.mkdir(exist_ok=True)
    clock = [1000.0]; delivered = []
    def transport(*args):
        def send(message):
            if clock[0] == 1000:
                raise TelegramDeliveryError('network', retryable=True)
            delivered.append(message)
        return send
    monkeypatch.setattr(module, 'load_credentials', lambda: SimpleNamespace(token='fixture', chat_id='fixture'))
    monkeypatch.setattr(module, 'TelegramTransport', transport)
    monkeypatch.setattr(module, '_rate_limiter', lambda run: None)
    monkeypatch.setattr(module, 'PersistentEventNotifier', lambda **kwargs:
                        PersistentEventNotifier(**kwargs, clock=lambda: clock[0]))
    with tempfile.TemporaryDirectory(dir=root) as temp:
        monkeypatch.setattr(module, 'ROOT', Path(temp))
        run = Path(temp) / 'isolated'
        directory = Path(temp) / 'results/diagnostics/quant_reports/isolated'
        directory.mkdir(parents=True)
        row = {'event_id': 'isolated-report', 'event_type': 'monitoring_alert',
               'payload': {'alert_type': 'TEST report', 'severity': 'INFO', 'status': 'RESOLVED'}}
        (directory / 'report_events.jsonl').write_text(json.dumps(row)+'\n')
        assert module._drain_quant_reports(run) == 'retry'
        clock[0] = 1006
        assert module._drain_quant_reports(run) == 'delivered'
        assert module._drain_quant_reports(run) == 'idle'
        assert len(delivered) == 1


def test_report_is_durably_queued_when_delivery_lock_is_busy(monkeypatch):
    import src.paper.quant_report as module
    from src.paper.single_writer import PaperWriterLock
    root = Path(__file__).resolve().parents[2] / '.test_scratch'
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(module, 'build_report', lambda *args, **kwargs: {
        'report_id': 'busy-report', 'period_start_utc': '2026-10-08T04:00:00Z',
        'period_end_utc_exclusive': '2026-10-09T04:00:00Z', 'period_has_ended': True,
        'metrics': {'trades': 0, 'net_pnl': 0.0}})
    with tempfile.TemporaryDirectory(dir=root) as temp:
        run = Path(temp) / 'run'; run.mkdir()
        output = Path(temp) / 'reports'
        directory = output / run.name
        with PaperWriterLock(directory / 'report_delivery.lock'):
            assert module.main(['--run-dir', str(run), '--date', '2026-10-08',
                                '--send', '--output-dir', str(output)]) == 0
            rows = (directory / 'report_events.jsonl').read_text().splitlines()
            assert len(rows) == 1
            assert json.loads(rows[0])['payload']['report_id'] == 'busy-report'
            assert not (directory / 'report_delivery_state.json').exists()
        assert list(run.iterdir()) == []


def test_default_calendar_reminder_requires_current_approved_identity(monkeypatch):
    import src.paper.notification_cli as module
    root = Path(__file__).resolve().parents[2] / '.test_scratch'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        monkeypatch.setattr(module, 'ROOT', Path(temp))
        monkeypatch.setattr(module, '_reviewed_calendar_for_run', lambda run:
                            SimpleNamespace(snapshot=SimpleNamespace(identity='approved')))
        config = Path(temp) / 'src/paper/config/paper_monitoring_maintenance.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'schema_version': 1, 'events': [{'snapshot_identity': 'old'}]}))
        assert module._default_maintenance_schedule(Path(temp)) is None
        config.write_text(json.dumps({'schema_version': 1, 'events': [{'snapshot_identity': 'approved'}]}))
        assert module._default_maintenance_schedule(Path(temp)) == config
