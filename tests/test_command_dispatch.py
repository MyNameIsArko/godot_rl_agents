import sys

from godot_rl import main as entrypoint
from godot_rl import project_cli


def test_project_doctor_reaches_project_commands(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["gdrl", "doctor", "--project", "game"])
    monkeypatch.setattr(project_cli, "main", lambda argv: seen.append(argv) or 1)
    assert entrypoint.main() == 1
    assert seen == [["doctor", "--project", "game"]]


def test_executable_doctor_keeps_legacy_arguments(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["gdrl", "doctor", "--env_path", "debug"])
    monkeypatch.setattr(entrypoint, "doctor", lambda argv: seen.append(argv) or 0)
    assert entrypoint.main() == 0
    assert seen == [["--env_path", "debug"]]
