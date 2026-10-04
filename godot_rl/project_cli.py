from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from typing import Any

import numpy as np

from godot_rl.core.project_env import GodotProjectEnv
from godot_rl.core.multi_agent_env import AGENT_IDS, GodotMultiAgentEnv

SCHEMA_VERSION = 1
SELF_PLAY_SCHEMA_VERSION = 2
DEFAULT_SCENE = "res://rl_training.tscn"
CONFIG_TEXT = """schema_version = 1
scene = "res://rl_training.tscn"
port = 11008
seed = 0
speedup = 8.0

[ppo]
n_steps = 64
batch_size = 64
learning_rate = 0.0003
ent_coef = 0.0001
clip_range = 0.2
"""
SELF_PLAY_CONFIG_TEXT = """schema_version = 2
mode = "self_play"
scene = "res://rl_training.tscn"
port = 11008
seed = 0
speedup = 8.0
agent_ids = ["player_0", "player_1"]

[ppo]
n_steps = 128
batch_size = 64
n_epochs = 10
learning_rate = 0.0003
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.0001
clip_range = 0.2
vf_coef = 0.5
max_grad_norm = 0.5

[self_play]
checkpoint_interval = 8192
"""
RL_PYPROJECT_TEXT = '''[project]\nname = "godot-rl-project"\nversion = "0.1.0"\nrequires-python = ">=3.8"\ndependencies = ["godot_rl"]\n'''
RL_GITIGNORE_TEXT = ".venv/\nmodels/\nruns/\n"
TOP_LEVEL_KEYS = {"schema_version", "scene", "port", "seed", "speedup", "ppo"}
PPO_KEYS = {"n_steps", "batch_size", "learning_rate", "ent_coef", "clip_range"}
SELF_PLAY_TOP_LEVEL_KEYS = {
    "schema_version", "mode", "scene", "port", "seed", "speedup", "agent_ids", "ppo", "self_play"
}
SELF_PLAY_PPO_KEYS = {
    "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "ent_coef",
    "clip_range", "vf_coef", "max_grad_norm"
}
SELF_PLAY_KEYS = {"checkpoint_interval"}
VERSION_RE = re.compile(r"\b(\d+)\.(\d+)(?:\.(\d+))?")


def _project_path(value: str | pathlib.Path) -> pathlib.Path:
    project = pathlib.Path(value).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"project directory does not exist: {project}")
    if not (project / "project.godot").is_file():
        raise ValueError(f"project.godot does not exist under: {project}")
    return project


def _scene_path(project: pathlib.Path, scene: Any) -> pathlib.Path:
    if not isinstance(scene, str) or not scene.startswith("res://") or not scene.endswith(".tscn"):
        raise ValueError("scene must be a res:// path ending in .tscn")
    relative = scene[len("res://"):]
    if not relative or "\\" in relative:
        raise ValueError("scene must stay inside res://")
    parts = pathlib.PurePosixPath(relative).parts
    if pathlib.PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ValueError("scene must stay inside res://")
    path = (project.joinpath(*parts)).resolve()
    try:
        path.relative_to(project)
    except ValueError as exc:
        raise ValueError("scene must stay inside res://") from exc
    if not path.is_file():
        raise ValueError(f"scene does not exist: {scene}")
    return path


def _validate_schema_v1(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != TOP_LEVEL_KEYS:
        unknown = sorted(set(config) - TOP_LEVEL_KEYS)
        missing = sorted(TOP_LEVEL_KEYS - set(config))
        raise ValueError(f"config keys differ: unknown={unknown}, missing={missing}")
    if isinstance(config["schema_version"], bool) or not isinstance(config["schema_version"], int):
        raise TypeError("schema_version must be an integer")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {config['schema_version']!r}")
    if not isinstance(config["ppo"], dict) or set(config["ppo"]) != PPO_KEYS:
        ppo = config["ppo"] if isinstance(config["ppo"], dict) else {}
        unknown = sorted(set(ppo) - PPO_KEYS)
        missing = sorted(PPO_KEYS - set(ppo))
        raise ValueError(f"ppo keys differ: unknown={unknown}, missing={missing}")
    if not isinstance(config["scene"], str):
        raise TypeError("scene must be a string")
    if isinstance(config["port"], bool) or not isinstance(config["port"], int) or not 1 <= config["port"] <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise TypeError("seed must be an integer")
    if isinstance(config["speedup"], bool) or not isinstance(config["speedup"], (int, float)):
        raise TypeError("speedup must be a finite positive number")
    if not math.isfinite(config["speedup"]) or config["speedup"] <= 0:
        raise ValueError("speedup must be a finite positive number")
    ppo = config["ppo"]
    for key in ("n_steps", "batch_size"):
        if isinstance(ppo[key], bool) or not isinstance(ppo[key], int) or ppo[key] <= 0:
            raise ValueError(f"ppo.{key} must be a positive integer")
    for key in ("learning_rate", "ent_coef", "clip_range"):
        if isinstance(ppo[key], bool) or not isinstance(ppo[key], (int, float)) or not math.isfinite(ppo[key]):
            raise ValueError(f"ppo.{key} must be finite")
    if ppo["learning_rate"] <= 0 or ppo["ent_coef"] < 0 or not 0 <= ppo["clip_range"] <= 1:
        raise ValueError("ppo values are outside their valid ranges")
    return config


def _positive_finite(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number")


def _validate_speedup_override(speedup: Any) -> None:
    if speedup is not None:
        _positive_finite(speedup, "speedup")


def _validate_schema_v2(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != SELF_PLAY_TOP_LEVEL_KEYS:
        unknown = sorted(set(config) - SELF_PLAY_TOP_LEVEL_KEYS)
        missing = sorted(SELF_PLAY_TOP_LEVEL_KEYS - set(config))
        raise ValueError(f"config keys differ: unknown={unknown}, missing={missing}")
    if config["mode"] != "self_play":
        raise ValueError("schema version 2 requires mode = 'self_play'")
    if config["agent_ids"] != ["player_0", "player_1"]:
        raise ValueError("agent_ids must be exactly ['player_0', 'player_1']")
    if not isinstance(config["ppo"], dict) or set(config["ppo"]) != SELF_PLAY_PPO_KEYS:
        ppo = config["ppo"] if isinstance(config["ppo"], dict) else {}
        unknown = sorted(set(ppo) - SELF_PLAY_PPO_KEYS)
        missing = sorted(SELF_PLAY_PPO_KEYS - set(ppo))
        raise ValueError(f"ppo keys differ: unknown={unknown}, missing={missing}")
    if not isinstance(config["self_play"], dict) or set(config["self_play"]) != SELF_PLAY_KEYS:
        self_play = config["self_play"] if isinstance(config["self_play"], dict) else {}
        unknown = sorted(set(self_play) - SELF_PLAY_KEYS)
        missing = sorted(SELF_PLAY_KEYS - set(self_play))
        raise ValueError(f"self_play keys differ: unknown={unknown}, missing={missing}")
    if not isinstance(config["scene"], str):
        raise TypeError("scene must be a string")
    if isinstance(config["port"], bool) or not isinstance(config["port"], int) or not 1 <= config["port"] <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise TypeError("seed must be an integer")
    _positive_finite(config["speedup"], "speedup")

    ppo = config["ppo"]
    for key in ("n_steps", "batch_size", "n_epochs"):
        if isinstance(ppo[key], bool) or not isinstance(ppo[key], int) or ppo[key] <= 0:
            raise ValueError(f"ppo.{key} must be a positive integer")
    for key in (
        "learning_rate", "gamma", "gae_lambda", "ent_coef", "clip_range", "vf_coef", "max_grad_norm"
    ):
        _positive_finite(ppo[key], f"ppo.{key}")
    if ppo["gamma"] > 1 or ppo["gae_lambda"] > 1 or ppo["clip_range"] > 1:
        raise ValueError("ppo.gamma, ppo.gae_lambda, and ppo.clip_range must not exceed 1")
    if ppo["batch_size"] > ppo["n_steps"] or ppo["n_steps"] % ppo["batch_size"]:
        raise ValueError("ppo.batch_size must divide ppo.n_steps")
    checkpoint_interval = config["self_play"]["checkpoint_interval"]
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
        or checkpoint_interval % ppo["n_steps"]
    ):
        raise ValueError("self_play.checkpoint_interval must be a positive multiple of ppo.n_steps")
    return config


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or "schema_version" not in config:
        raise ValueError("config must contain schema_version")
    schema_version = config["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("schema_version must be an integer")
    if schema_version == SCHEMA_VERSION:
        return _validate_schema_v1(config)
    if schema_version == SELF_PLAY_SCHEMA_VERSION:
        return _validate_schema_v2(config)
    raise ValueError(f"unsupported config schema_version: {schema_version!r}")


def read_config(project: pathlib.Path) -> dict[str, Any]:
    path = project / "rl" / "config.toml"
    try:
        with path.open("rb") as file:
            return _validate_config(tomllib.load(file))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc


def _addon_files(addon_path: str | pathlib.Path | None) -> list[tuple[pathlib.Path, pathlib.Path]]:
    if addon_path is None:
        return []
    addon = pathlib.Path(addon_path).expanduser().resolve()
    if (addon / "addons/godot_rl_agents").is_dir():
        addon = addon / "addons/godot_rl_agents"
    if not (addon / "sync.gd").is_file() or not (addon / "plugin.cfg").is_file():
        raise ValueError("addon-path must contain the Godot RL Agents add-on")
    files = []
    for root, directories, filenames in os.walk(addon, followlinks=False):
        for name in directories + filenames:
            path = pathlib.Path(root) / name
            if path.is_symlink():
                raise ValueError(f"add-on contains a symlink: {path}")
        for name in filenames:
            if name.endswith((".uid", ".import")):
                continue
            path = pathlib.Path(root) / name
            files.append((path, pathlib.Path("addons/godot_rl_agents") / path.relative_to(addon)))
    license_path = addon.parents[1] / "LICENSE"
    if license_path.is_file():
        files.append((license_path, pathlib.Path("addons/godot_rl_agents/LICENSE")))
    return files


def init_project(
    project_value: str | pathlib.Path, scene: str, mode: str | None = None,
    addon_path: str | pathlib.Path | None = None,
    package_source: str | pathlib.Path | None = None, sync: bool = True,
) -> int:
    project = _project_path(project_value)
    _scene_path(project, scene)
    if mode not in (None, "self-play"):
        raise ValueError("mode must be self-play when provided")
    template = SELF_PLAY_CONFIG_TEXT if mode == "self-play" else CONFIG_TEXT
    pyproject = RL_PYPROJECT_TEXT
    if package_source is not None:
        package = pathlib.Path(package_source).expanduser().resolve()
        if not (package / "pyproject.toml").is_file():
            raise ValueError("package-source must contain pyproject.toml")
        pyproject = pyproject.replace('"godot_rl"', json.dumps("godot_rl @ " + package.as_uri()))
    planned = {
        pathlib.Path("rl/config.toml"): template.replace(DEFAULT_SCENE, scene).encode(),
        pathlib.Path("rl/pyproject.toml"): pyproject.encode(),
        pathlib.Path("rl/.gitignore"): RL_GITIGNORE_TEXT.encode(),
    }
    for source, relative in _addon_files(addon_path):
        planned[relative] = source.read_bytes()
    differences = []
    for relative, content in planned.items():
        target = project / relative
        parent = project
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent.is_symlink() or parent.exists() and not parent.is_dir():
                differences.append(str(parent.relative_to(project)))
                break
        if target.is_symlink() or target.exists() and (not target.is_file() or target.read_bytes() != content):
            differences.append(str(relative))
    if differences:
        raise ValueError("managed files differ:\n" + "\n".join(sorted(set(differences))))
    for relative, content in planned.items():
        target = project / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as file:
            file.write(content)
    if sync:
        subprocess.run(["uv", "sync", "--project", str(project / "rl")], shell=False, check=True)
    print(f"Initialized {project}")
    return 0


def _godot_candidates(explicit: str | None) -> list[pathlib.Path]:
    candidates = []
    if explicit:
        candidates.append(pathlib.Path(explicit).expanduser())
    if os.environ.get("GODOT_BIN"):
        candidates.append(pathlib.Path(os.environ["GODOT_BIN"]).expanduser())
    for name in ("godot", "godot4"):
        path = shutil.which(name)
        if path:
            candidates.append(pathlib.Path(path))
    if platform.system() == "Darwin":
        candidates.extend(
            [
                pathlib.Path("/Applications/Godot.app/Contents/MacOS/Godot"),
                pathlib.Path("/Applications/Godot_mono.app/Contents/MacOS/Godot"),
            ]
        )
    elif platform.system() == "Windows":
        candidates.extend(
            [
                pathlib.Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Godot/Godot.exe",
                pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Godot/Godot.exe",
            ]
        )
    return candidates


def discover_godot(explicit: str | None = None) -> pathlib.Path | None:
    for candidate in _godot_candidates(explicit):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _godot_version(path: pathlib.Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True, shell=False, check=False)
    except OSError as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip() or f"exited with code {result.returncode}"
    match = VERSION_RE.search(result.stdout)
    if not match:
        return None, "could not parse Godot version"
    version = ".".join(part for part in match.groups() if part is not None)
    return version, None


def _python_executable(rl: pathlib.Path) -> pathlib.Path:
    return rl / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _imports(python: pathlib.Path) -> tuple[dict[str, str], str | None]:
    code = "import gymnasium, stable_baselines3; print(gymnasium.__version__); print(stable_baselines3.__version__)"
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, shell=False, check=False)
    except OSError as exc:
        return {}, str(exc)
    if result.returncode:
        return {}, result.stderr.strip() or f"exited with code {result.returncode}"
    versions = result.stdout.splitlines()
    if len(versions) < 2:
        return {}, "import check returned incomplete versions"
    return {"gymnasium": versions[-2].strip(), "stable_baselines3": versions[-1].strip()}, None


def _port_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def doctor(project_value: str | pathlib.Path, godot: str | None = None) -> dict[str, Any]:
    project = pathlib.Path(project_value).expanduser().resolve()
    result: dict[str, Any] = {
        "ok": False,
        "project": str(project),
        "scene": None,
        "godot": {"ok": False, "path": None, "version": None},
        "plugin": {"ok": False, "version": None},
        "python": {"ok": False, "version": None},
        "imports": {},
        "problems": [],
    }
    problems: list[str] = result["problems"]
    if not (project / "project.godot").is_file():
        problems.append("project.godot is missing")

    config: dict[str, Any] | None = None
    try:
        config = read_config(project)
        result["scene"] = config["scene"]
        _scene_path(project, config["scene"])
    except (OSError, TypeError, ValueError) as exc:
        problems.append(str(exc))

    if config is not None and config.get("schema_version") == SELF_PLAY_SCHEMA_VERSION:
        result.update({"mode": "self_play", "protocol": {"major": 2, "minor": 0},
                       "agents": ["player_0", "player_1"]})
    addon = project / "addons/godot_rl_agents"
    manifest = addon / "plugin.cfg"
    text = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version = version_match.group(1) if version_match else None
    sync_path = addon / "sync.gd"
    sync_text = sync_path.read_text(encoding="utf-8") if sync_path.is_file() else ""
    needed = "PROTOCOL_TWO_MAJOR" if config and config.get("schema_version") == 2 else "PROJECT_PROTOCOL_MAJOR"
    plugin_ok = version is not None and needed in sync_text
    result["plugin"] = {"ok": plugin_ok, "version": version}
    if not plugin_ok:
        problems.append("Install the compatible project protocol add-on from its separate repository")

    godot_path = discover_godot(godot)
    if godot_path is None:
        problems.append("Godot executable was not found")
    else:
        version, error = _godot_version(godot_path)
        result["godot"] = {"ok": False, "path": str(godot_path), "version": version}
        if error:
            problems.append(f"Godot version check failed: {error}")
        elif version is None:
            problems.append("Godot version check returned no version")
        else:
            numbers = tuple(int(part) for part in version.split("."))
            result["godot"]["ok"] = numbers >= (4, 3)
            if not result["godot"]["ok"]:
                problems.append(f"Godot {version} is below the 4.3 floor")

    rl = project / "rl"
    python = _python_executable(rl)
    if not python.is_file():
        python = pathlib.Path(sys.executable)
    if not python.is_file():
        problems.append(f"project virtual environment is missing: {python}")
    else:
        try:
            python_result = subprocess.run([str(python), "--version"], capture_output=True, text=True, shell=False, check=False)
            version_match = VERSION_RE.search(python_result.stdout + python_result.stderr)
            python_version = version_match.group(0) if version_match else None
        except OSError:
            python_version = None
        python_numbers = tuple(int(part) for part in python_version.split(".")) if python_version else ()
        python_ok = python_version is not None and python_numbers >= (3, 8)
        result["python"] = {"ok": python_ok, "version": python_version}
        if python_version is None:
            problems.append("Python version check failed")
        elif not python_ok:
            problems.append(f"Python {python_version} is outside the supported range")
        else:
            imports, error = _imports(python)
            result["imports"] = imports
            if error:
                problems.append(f"Python imports failed: {error}")
            else:
                if not imports.get("gymnasium") or not imports.get("stable_baselines3"):
                    problems.append("Gymnasium or Stable Baselines3 is unavailable")

    if config is not None and not _port_free(config["port"]):
        problems.append(f"TCP port is not free: {config['port']}")
    result["ok"] = not problems
    return result


def validate_project(project_value: str | pathlib.Path, steps: int, godot: str | None = None) -> int:
    if steps <= 0:
        raise ValueError("steps must be positive")
    project = _project_path(project_value)
    report = doctor(project, godot)
    if not report["ok"]:
        print(json.dumps(report, sort_keys=True))
        return 1
    config = read_config(project)
    if config["schema_version"] == SELF_PLAY_SCHEMA_VERSION:
        return _validate_self_play_project(project, config, steps, report)

    environment: GodotProjectEnv | None = None
    try:
        from stable_baselines3.common.env_checker import check_env

        environment = GodotProjectEnv(
            godot_path=pathlib.Path(report["godot"]["path"]),
            project_path=project,
            scene=config["scene"],
            port=config["port"],
            seed=config["seed"],
            speedup=config["speedup"],
        )
        check_env(environment, skip_render_check=True)
        environment.reset(seed=config["seed"])
        for _ in range(steps):
            _, _, terminated, truncated, _ = environment.step(environment.action_space.sample())
            if terminated or truncated:
                environment.reset(seed=config["seed"])
    finally:
        if environment is not None:
            environment.close()
    print(json.dumps({"ok": True, "steps": steps}, sort_keys=True))
    return 0


def _finite_value(value: Any) -> bool:
    if isinstance(value, numbers.Real):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_value(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_value(child) for child in value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return True
    return array.dtype.kind not in "fiu" or bool(np.isfinite(array).all())


def _require_agent_mapping(value: Any, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"player_0", "player_1"}:
        raise ValueError(f"{field} must contain exactly player_0 and player_1")


def _validate_self_play_project(
    project: pathlib.Path,
    config: dict[str, Any],
    steps: int,
    report: dict[str, Any],
) -> int:
    environment: GodotMultiAgentEnv | None = None
    episodes = 0
    try:
        environment = GodotMultiAgentEnv(
            godot_path=pathlib.Path(report["godot"]["path"]),
            project_path=project,
            scene=config["scene"],
            port=config["port"],
            seed=config["seed"],
            speedup=config["speedup"],
        )
        if any(
            environment.observation_spaces[AGENT_IDS[0]] != environment.observation_spaces[agent_id]
            for agent_id in AGENT_IDS[1:]
        ) or any(
            environment.action_spaces[AGENT_IDS[0]] != environment.action_spaces[agent_id]
            for agent_id in AGENT_IDS[1:]
        ):
            raise ValueError("self-play observation and action spaces must match")
        observations, infos = environment.reset(seed=config["seed"])
        _require_agent_mapping(observations, "reset observations")
        _require_agent_mapping(infos, "reset infos")
        for agent_id in AGENT_IDS:
            if not _finite_value(observations[agent_id]):
                raise ValueError(f"reset.observations.{agent_id} contains a non-finite value")
            if not environment.observation_spaces[agent_id].contains(observations[agent_id]):
                raise ValueError(f"reset.observations.{agent_id} is outside its space")

        for step in range(steps):
            actions = {
                agent_id: environment.action_spaces[agent_id].sample()
                for agent_id in AGENT_IDS
            }
            for agent_id, action in actions.items():
                if not _finite_value(action) or not environment.action_spaces[agent_id].contains(action):
                    raise ValueError(f"step {step}.actions.{agent_id} is invalid")
            observations, rewards, terminated, truncated, infos = environment.step(actions)
            for field, value in (
                ("observations", observations),
                ("rewards", rewards),
                ("terminated", terminated),
                ("truncated", truncated),
                ("infos", infos),
            ):
                _require_agent_mapping(value, f"step {step} {field}")
            if set(terminated.values()) != {True} and set(terminated.values()) != {False}:
                raise ValueError(f"step {step}.terminated flags must stay synchronized")
            if set(truncated.values()) != {True} and set(truncated.values()) != {False}:
                raise ValueError(f"step {step}.truncated flags must stay synchronized")
            if any(terminated.values()) and any(truncated.values()):
                raise ValueError(f"step {step} cannot be terminated and truncated together")
            for agent_id in AGENT_IDS:
                if not _finite_value(observations[agent_id]) or not environment.observation_spaces[agent_id].contains(observations[agent_id]):
                    raise ValueError(f"step {step}.observations.{agent_id} is invalid")
                if isinstance(rewards[agent_id], bool) or not isinstance(rewards[agent_id], numbers.Real) or not math.isfinite(rewards[agent_id]):
                    raise ValueError(f"step {step}.rewards.{agent_id} must be finite")
                if not isinstance(terminated[agent_id], bool) or not isinstance(truncated[agent_id], bool):
                    raise TypeError(f"step {step}.flags.{agent_id} must be boolean")
                if not _finite_value(infos[agent_id]):
                    raise ValueError(f"step {step}.infos.{agent_id} contains a non-finite value")
            if any(terminated.values()) or any(truncated.values()):
                outcomes = [infos[agent_id].get("outcome") for agent_id in AGENT_IDS]
                if outcomes not in (["win", "loss"], ["loss", "win"], ["draw", "draw"]):
                    raise ValueError(f"step {step}.info.outcome values are not complementary")
                episodes += 1
                observations, infos = environment.reset(seed=config["seed"])
                _require_agent_mapping(observations, "reset observations")
                _require_agent_mapping(infos, "reset infos")
        if episodes == 0:
            raise ValueError(
                "validation ended before a completed episode; increase --steps or reduce the scene episode limit"
            )
    finally:
        if environment is not None:
            environment.close()
    print(json.dumps({"ok": True, "steps": steps, "episodes": episodes}, sort_keys=True))
    return 0


def train_project(
    project_value: str | pathlib.Path,
    timesteps: int,
    name: str,
    resume: str | None = None,
    show_window: bool = False,
    seed: int | None = None,
    speedup: float | None = None,
    torch_threads: int = 1,
    godot: str | None = None,
) -> int:
    _validate_speedup_override(speedup)
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("name must be a single safe path component")
    if timesteps <= 0 or torch_threads <= 0:
        raise ValueError("timesteps and torch-threads must be positive")
    project = _project_path(project_value)
    try:
        parsed_config = read_config(project)
    except ValueError:
        parsed_config = None
    if parsed_config is not None and parsed_config.get("schema_version") == SELF_PLAY_SCHEMA_VERSION:
        raise ValueError("gdrl train supports schema version 1 only; use gdrl self-play")
    report = doctor(project, godot)
    if not report["ok"]:
        print(json.dumps(report, sort_keys=True))
        return 1
    config = read_config(project)
    model_path = project / "rl" / "models" / f"{name}.zip"
    if resume is None and model_path.exists():
        raise ValueError(f"model already exists: {model_path}")
    resume_path = pathlib.Path(resume).expanduser().resolve() if resume else None
    if resume_path is not None and not resume_path.is_file():
        raise ValueError(f"resume model does not exist: {resume_path}")
    environment: GodotProjectEnv | None = None
    vector_environment: Any = None
    model: Any = None
    run_dir = project / "rl" / "runs" / name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(torch_threads)
        from godot_rl.wrappers.project_sb3 import GodotProjectVecEnv

        environment = GodotProjectEnv(
            godot_path=pathlib.Path(report["godot"]["path"]),
            project_path=project,
            scene=config["scene"],
            port=config["port"],
            seed=config["seed"] if seed is None else seed,
            speedup=config["speedup"] if speedup is None else speedup,
            show_window=show_window,
        )
        vector_environment = GodotProjectVecEnv(environment)
        ppo = config["ppo"]
        if resume_path is None:
            model_kwargs = {
                "n_steps": ppo["n_steps"],
                "batch_size": ppo["batch_size"],
                "learning_rate": ppo["learning_rate"],
                "ent_coef": ppo["ent_coef"],
                "clip_range": ppo["clip_range"],
                "seed": config["seed"] if seed is None else seed,
            }
            try:
                __import__("tensorboard")
            except ImportError:
                pass
            else:
                model_kwargs["tensorboard_log"] = str(run_dir)
            model = PPO(
                "MultiInputPolicy",
                vector_environment,
                **model_kwargs,
            )
        else:
            model = PPO.load(str(resume_path), env=vector_environment)
        try:
            model.learn(total_timesteps=timesteps)
        except KeyboardInterrupt:
            model.save(str(model_path.with_suffix(".interrupted")))
            raise
        model.save(str(model_path.with_suffix("")))
    finally:
        if vector_environment is not None:
            vector_environment.close()
        elif environment is not None:
            environment.close()
    print(f"Saved model to {model_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gdrl")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--project", required=True)
    init.add_argument("--scene", required=True)
    init.add_argument("--mode", choices=("self-play",))
    init.add_argument("--addon-path")
    init.add_argument("--package-source")
    init.add_argument("--no-sync", action="store_true")
    for command in ("doctor", "validate", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--project", required=True)
        sub.add_argument("--godot")
        if command == "validate":
            sub.add_argument("--steps", type=int, default=32)
        if command == "train":
            sub.add_argument("--timesteps", type=int, required=True)
            sub.add_argument("--name", required=True)
            sub.add_argument("--resume")
            sub.add_argument("--show-window", action="store_true")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--speedup", type=float)
            sub.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return init_project(args.project, args.scene, args.mode, args.addon_path, args.package_source, not args.no_sync)
        if args.command == "doctor":
            report = doctor(args.project, args.godot)
            print(json.dumps(report, sort_keys=True))
            return 0 if report["ok"] else 1
        if args.command == "validate":
            return validate_project(args.project, args.steps, args.godot)
        return train_project(args.project, args.timesteps, args.name, args.resume,
                             args.show_window, args.seed, args.speedup, args.torch_threads, args.godot)
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"gdrl: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
