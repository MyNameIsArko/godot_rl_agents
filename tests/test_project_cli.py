import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

from godot_rl import project_cli as cli


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n")
    (project / "rl_training.tscn").write_text("[gd_scene format=3]\n")
    return project


def test_train_rejects_unsafe_names(tmp_path):
    for name in ("nested/run", r"nested\run", ".."):
        with pytest.raises(ValueError, match="safe path"):
            cli.train_project(tmp_path, 1, name)


def test_train_saves_interrupted_model_and_closes_environment(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    (project / "rl").mkdir()
    (project / "rl/config.toml").write_text(cli.CONFIG_TEXT)
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {
        "ok": True,
        "godot": {"path": str(tmp_path / "Godot")},
    })
    class Raw:
        def close(self):
            self.closed = True

    raw = Raw()
    class Vec:
        def __init__(self, environment):
            self.environment = environment
        def close(self):
            raw.close()

    class Model:
        def learn(self, total_timesteps):
            raise KeyboardInterrupt
        def save(self, path):
            Path(path + ".zip").touch()

    class FakePPO:
        def __new__(cls, *args, **kwargs):
            return Model()

    monkeypatch.setattr(cli, "GodotProjectEnv", lambda **kwargs: raw)

    import stable_baselines3

    from godot_rl.wrappers import project_sb3 as sb3_module
    monkeypatch.setattr(sb3_module, "GodotProjectVecEnv", Vec)
    monkeypatch.setattr(stable_baselines3, "PPO", FakePPO)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)

    with pytest.raises(KeyboardInterrupt):
        cli.train_project(project, 1, "interrupted", torch_threads=1)
    assert raw.closed is True
    assert (project / "rl/models/interrupted.interrupted.zip").is_file()


def test_train_omits_tensorboard_log_when_tensorboard_is_absent(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    (project / "rl").mkdir()
    (project / "rl/config.toml").write_text(cli.CONFIG_TEXT)
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {
        "ok": True,
        "godot": {"path": str(tmp_path / "Godot")},
    })
    monkeypatch.setitem(sys.modules, "tensorboard", None)
    seen = {}

    class Raw:
        def close(self):
            pass

    class Vec:
        def __init__(self, environment):
            self.environment = environment

        def close(self):
            pass

    class Model:
        def learn(self, total_timesteps):
            return self

        def save(self, path):
            Path(path + ".zip").touch()

    class FakePPO:
        def __new__(cls, *args, **kwargs):
            seen.update(kwargs)
            return Model()

    monkeypatch.setattr(cli, "GodotProjectEnv", lambda **kwargs: Raw())
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    import stable_baselines3

    from godot_rl.wrappers import project_sb3 as sb3_module
    monkeypatch.setattr(sb3_module, "GodotProjectVecEnv", Vec)
    monkeypatch.setattr(stable_baselines3, "PPO", FakePPO)

    assert cli.train_project(project, 1, "without-tensorboard", torch_threads=1) == 0
    assert "tensorboard_log" not in seen
    assert (project / "rl/runs/without-tensorboard").is_dir()



def make_addon(tmp_path):
    addon = tmp_path / "addon"
    addon.mkdir()
    (addon / "plugin.cfg").write_text('[plugin]\nversion="0.8"\n')
    (addon / "sync.gd").write_text("const PROJECT_PROTOCOL_MAJOR = 1\nconst PROTOCOL_TWO_MAJOR = 2\n")
    return addon


def test_init_preserves_addon_and_is_safe_to_repeat(tmp_path):
    project = make_project(tmp_path)
    addon = make_addon(tmp_path)
    (addon / "onnx").mkdir()
    (addon / "onnx/existing.gd").write_text("# existing upstream feature")
    cli.init_project(project, "res://rl_training.tscn", addon_path=addon, sync=False)
    files = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    cli.init_project(project, "res://rl_training.tscn", addon_path=addon, sync=False)
    assert files == {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    assert (project / "addons/godot_rl_agents/onnx/existing.gd").is_file()
    assert "godot_rl" in (project / "rl/pyproject.toml").read_text()


def test_init_refuses_modified_files_before_any_write(tmp_path):
    project = make_project(tmp_path)
    cli.init_project(project, "res://rl_training.tscn", sync=False)
    (project / "rl/config.toml").write_text("custom configuration")
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="managed files differ"):
        cli.init_project(project, "res://rl_training.tscn", addon_path=make_addon(tmp_path), sync=False)
    assert before == {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}


def test_init_refuses_parent_symlink(tmp_path):
    project = make_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "rl").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="managed files differ"):
        cli.init_project(project, "res://rl_training.tscn", sync=False)
    assert list(outside.iterdir()) == []


def test_init_accepts_separate_python_source_and_self_play(tmp_path):
    project = make_project(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "pyproject.toml").write_text("[project]")
    cli.init_project(project, "res://rl_training.tscn", "self-play", package_source=package, sync=False)
    assert cli.read_config(project)["agent_ids"] == ["player_0", "player_1"]
    assert package.as_uri() in (project / "rl/pyproject.toml").read_text()


def test_doctor_reports_installed_addon_and_current_python(tmp_path, monkeypatch, capsys):
    project = make_project(tmp_path)
    addon = make_addon(tmp_path)
    cli.init_project(project, "res://rl_training.tscn", addon_path=addon, sync=False)
    capsys.readouterr()
    godot = tmp_path / "Godot"
    godot.touch()
    monkeypatch.setattr(cli, "_godot_version", lambda path: ("4.7.2", None))
    monkeypatch.setattr(cli, "_imports", lambda python: ({"gymnasium": "1.0.0", "stable_baselines3": "2.4.0"}, None))
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "Python 3.13.2", ""))
    assert cli.main(["doctor", "--project", str(project), "--godot", str(godot)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["plugin"]["version"] == "0.8"


def test_doctor_reports_missing_components(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "discover_godot", lambda value: None)
    report = cli.doctor(tmp_path)
    assert not report["ok"]
    assert any("project.godot" in problem for problem in report["problems"])
    assert any("add-on" in problem for problem in report["problems"])
    assert any("Godot executable" in problem for problem in report["problems"])
