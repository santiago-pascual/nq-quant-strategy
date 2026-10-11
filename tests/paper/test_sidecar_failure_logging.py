import json
from src.paper.notification_cli import _log_sidecar_failure


def test_error_message_and_url_secrets_are_never_logged(capsys):
    secret='never-log-this-token'
    try:
        raise PermissionError('https://api.telegram.org/bot'+secret+'/sendMessage')
    except PermissionError as exc:
        _log_sidecar_failure(exc)
    logged=capsys.readouterr().err
    assert secret not in logged and 'https://' not in logged
    record=json.loads(logged)
    assert record['error_type']=='PermissionError'
    assert record['frames'][-1]['function']=='test_error_message_and_url_secrets_are_never_logged'
    assert 'line' in record['frames'][-1]
