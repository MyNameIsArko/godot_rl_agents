import json
import socket

from godot_rl.main import _doctor_report


def test_doctor_reports_a_valid_debug_environment(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    report = _doctor_report("debug", port)

    assert report["environment"] == {"ok": True, "path": "debug"}
    assert report["port"]["ok"] is True
    assert all(report["dependencies"].values())
    assert report["ok"] is True


def test_doctor_rejects_missing_environment(tmp_path):
    report = _doctor_report(str(tmp_path / "missing"), 0)

    assert report["environment"]["ok"] is False
    assert report["ok"] is False
