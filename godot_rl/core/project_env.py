from __future__ import annotations

import atexit
import math
import os
import pathlib
import signal
import socket
import subprocess
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from godot_rl.core.protocol import (
    CONNECTION_TIMEOUT,
    DEFAULT_HOST,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    READ_TIMEOUT,
    ProtocolError,
    make_handshake,
    recv_frame,
    send_frame,
    validate_finite,
    validate_handshake,
)


class _GodotProcessSession:
    """Own one Godot child process and its protocol connection."""

    def __init__(
        self,
        *,
        godot_path: pathlib.Path,
        project_path: pathlib.Path,
        scene: str,
        port: int = 11008,
        seed: int = 0,
        speedup: float = 8.0,
        show_window: bool = False,
        protocol_major: int = PROTOCOL_MAJOR,
    ) -> None:
        self.godot_path = pathlib.Path(godot_path).expanduser().resolve()
        self.project_path = pathlib.Path(project_path).expanduser().resolve()
        self.scene = scene
        self.port = port
        self.seed_value = seed
        self.speedup = speedup
        self.show_window = show_window
        self.protocol_major = protocol_major
        self.process: subprocess.Popen[bytes] | None = None
        self._listener: socket.socket | None = None
        self.connection: socket.socket | None = None
        self._closed = False

        self._validate_configuration()
        try:
            self._listener = self._bind_listener()
            self.process = self._launch_godot()
            self.connection, _ = self._listener.accept()
            self.connection.settimeout(READ_TIMEOUT)
            self._listener.close()
            self._listener = None
            self.handshake()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _validate_configuration(self) -> None:
        if not isinstance(self.protocol_major, int) or isinstance(self.protocol_major, bool):
            raise TypeError("protocol_major must be an integer")
        if not self.godot_path.is_file():
            raise FileNotFoundError(f"Godot executable does not exist: {self.godot_path}")
        if not self.project_path.is_dir():
            raise FileNotFoundError(f"Godot project does not exist: {self.project_path}")
        if not (self.project_path / "project.godot").is_file():
            raise FileNotFoundError(f"project.godot does not exist under: {self.project_path}")
        if not isinstance(self.scene, str) or not self.scene.startswith("res://") or not self.scene.endswith(".tscn"):
            raise ValueError("scene must be a res:// path ending in .tscn")
        scene_path = pathlib.PurePosixPath(self.scene[len("res://"):])
        if scene_path.is_absolute() or ".." in scene_path.parts:
            raise ValueError("scene must stay inside res://")
        resolved_scene = (self.project_path / pathlib.Path(*scene_path.parts)).resolve()
        try:
            resolved_scene.relative_to(self.project_path)
        except ValueError as exc:
            raise ValueError("scene must stay inside res://") from exc
        if not resolved_scene.is_file():
            raise FileNotFoundError(f"scene does not exist: {self.scene}")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not isinstance(self.seed_value, int) or isinstance(self.seed_value, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(self.speedup, (int, float)) or isinstance(self.speedup, bool):
            raise TypeError("speedup must be a finite positive number")
        if not math.isfinite(self.speedup) or self.speedup <= 0:
            raise ValueError("speedup must be a finite positive number")

    def _bind_listener(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((DEFAULT_HOST, self.port))
            listener.listen(1)
            listener.settimeout(CONNECTION_TIMEOUT)
        except BaseException:
            listener.close()
            raise
        return listener

    def _build_launch_command(self) -> list[str]:
        command = [
            str(self.godot_path),
            "--path",
            str(self.project_path),
        ]
        if not self.show_window:
            command.extend(["--headless", "--disable-render-loop"])
        command.extend(
            [
                "--scene",
                self.scene,
                f"--port={self.port}",
                f"--env_seed={self.seed_value}",
                f"--speedup={self.speedup}",
            ]
        )
        return command

    def _launch_godot(self) -> subprocess.Popen[bytes]:
        kwargs: dict[str, Any] = {"shell": False}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            if not self.show_window:
                kwargs["creationflags"] |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return subprocess.Popen(self._build_launch_command(), **kwargs)

    def _require_connection(self) -> socket.socket:
        if self.connection is None or self._closed:
            raise RuntimeError("Godot process session is closed")
        return self.connection

    def handshake(self) -> None:
        connection = self._require_connection()
        send_frame(connection, make_handshake(self.protocol_major))
        response = recv_frame(connection)
        validate_handshake(response, self.protocol_major)

    def send(self, message: dict[str, Any]) -> None:
        send_frame(self._require_connection(), message)

    def receive(self) -> dict[str, Any]:
        return recv_frame(self._require_connection())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "process", None)
        if self.connection is not None:
            connection = self.connection
            self.connection = None
            try:
                connection.settimeout(2.0)
                send_frame(connection, {"type": "close"})
            except (OSError, ProtocolError):
                pass
            finally:
                connection.close()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if process is not None and process.poll() is None and not _wait_for_process(process, 2.0):
            _terminate_process(process)
            if not _wait_for_process(process, 2.0):
                _kill_process(process)
                _wait_for_process(process, 2.0)
        try:
            atexit.unregister(self.close)
        except ValueError:
            pass


class GodotProjectEnv(gym.Env):
    """A one-agent Gymnasium environment backed by a Godot game process."""

    metadata: ClassVar = {"render_modes": []}

    def __init__(
        self,
        *,
        godot_path: pathlib.Path,
        project_path: pathlib.Path,
        scene: str,
        port: int = 11008,
        seed: int = 0,
        speedup: float = 8.0,
        show_window: bool = False,
    ) -> None:
        self.godot_path = pathlib.Path(godot_path).expanduser().resolve()
        self.project_path = pathlib.Path(project_path).expanduser().resolve()
        self.scene = scene
        self.port = port
        self.seed_value = seed
        self.speedup = speedup
        self.show_window = show_window
        self._closed = False

        try:
            self._session = _GodotProcessSession(
                godot_path=self.godot_path,
                project_path=self.project_path,
                scene=self.scene,
                port=self.port,
                seed=self.seed_value,
                speedup=self.speedup,
                show_window=self.show_window,
                protocol_major=PROTOCOL_MAJOR,
            )
            self.process = self._session.process
            self._listener = self._session._listener
            self.connection = self._session.connection
            self._read_env_info()
        except BaseException:
            self.close()
            raise

    def _require_connection(self) -> socket.socket:
        if self._closed:
            raise RuntimeError("GodotProjectEnv is closed")
        return self._session._require_connection()

    def _handshake(self) -> None:
        self._session.handshake()

    def _read_env_info(self) -> None:
        self._session.send(
            {"type": "env_info", "protocol": {"major": PROTOCOL_MAJOR, "minor": PROTOCOL_MINOR}},
        )
        response = self._session.receive()
        if response.get("type") != "env_info":
            raise ProtocolError("expected env_info response")
        protocol = response.get("protocol")
        if not isinstance(protocol, dict):
            raise ProtocolError("env_info is missing protocol version")
        if protocol.get("major") != PROTOCOL_MAJOR:
            raise ProtocolError(
                f"protocol major mismatch: peer={protocol.get('major')}, expected={PROTOCOL_MAJOR}"
            )
        if response.get("agent_count") != 1:
            raise ValueError("GodotProjectEnv supports exactly one agent")
        self.observation_space = _space_from_agent_spec(response["observation_space"], observation=True)
        self.action_space = _space_from_agent_spec(response["action_space"], observation=False)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        message: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            message["seed"] = int(seed)
        self._session.send(message)
        response = self._session.receive()
        if response.get("type") != "reset":
            raise ProtocolError("expected reset response")
        observation = self._coerce_observation(response.get("observation", response.get("obs")))
        info = self._coerce_info(response.get("info", {}))
        return observation, info

    def step(self, action: Any):
        self.step_send(action)
        return self.step_recv()

    def step_send(self, action: Any) -> None:
        action = self._coerce_value(action, self.action_space, "action")
        self._session.send({"type": "step", "action": _json_value(action)})

    def step_recv(self):
        response = self._session.receive()
        if response.get("type") != "step":
            raise ProtocolError("expected step response")
        if "done" in response:
            raise ProtocolError("protocol version 1 step response must not contain done")
        terminated = response.get("terminated")
        truncated = response.get("truncated")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise ProtocolError("step response must contain boolean terminated and truncated fields")
        observation = self._coerce_observation(response.get("observation", response.get("obs")))
        reward = response.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("reward must be a finite number")
        info = self._coerce_info(response.get("info", {}))
        return observation, float(reward), terminated, truncated, info

    def _coerce_observation(self, value: Any) -> Any:
        return self._coerce_value(value, self.observation_space, "observation")

    def _coerce_info(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("information value must be a dictionary")
        validate_finite(value, "information")
        return value

    def _coerce_value(self, value: Any, space: gym.Space, field: str) -> Any:
        validate_finite(value, field)
        if isinstance(space, gym.spaces.Dict):
            if not isinstance(value, dict):
                raise TypeError(f"{field} must be a dictionary")
            missing = set(space.spaces) - set(value)
            if missing:
                raise ValueError(f"{field} is missing field '{min(missing)}'")
            return {key: self._coerce_value(value[key], child, f"{field}.{key}") for key, child in space.spaces.items()}
        if isinstance(space, gym.spaces.Box):
            try:
                value = np.asarray(value, dtype=space.dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} is invalid: {exc}") from exc
        if not space.contains(value):
            raise ValueError(f"{field} is outside the declared space")
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()


def _wait_for_process(process: subprocess.Popen[Any], timeout: float) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        process.terminate()
        return
    _signal_posix_process(process, signal.SIGTERM)


def _kill_process(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        process.kill()
        return
    _signal_posix_process(process, signal.SIGKILL)


def _signal_posix_process(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (AttributeError, OSError, ProcessLookupError):
        process.terminate() if sig == signal.SIGTERM else process.kill()


def _space_from_agent_spec(spec: Any, *, observation: bool) -> gym.Space:
    if isinstance(spec, list):
        if len(spec) != 1:
            raise ValueError("GodotProjectEnv supports one agent space")
        spec = spec[0]
    if isinstance(spec, gym.Space):
        return spec
    if not isinstance(spec, dict):
        raise TypeError("space declaration must be a dictionary")
    if spec.get("type") == "dict" or spec.get("space") == "dict":
        return gym.spaces.Dict({key: _space_from_spec(value, observation=observation) for key, value in spec["spaces"].items()})
    if all(isinstance(value, dict) and ("space" in value or "action_type" in value) for value in spec.values()):
        return gym.spaces.Dict({key: _space_from_spec(value, observation=observation) for key, value in spec.items()})
    return _space_from_spec(spec, observation=observation)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _space_from_spec(spec: dict[str, Any], *, observation: bool) -> gym.Space:
    kind = spec.get("space", spec.get("type", spec.get("action_type")))
    if kind in {"box", "continuous"}:
        raw_shape = spec.get("shape", spec.get("size", (1,)))
        shape = (raw_shape,) if isinstance(raw_shape, int) else tuple(raw_shape)
        low = spec.get("low", -1.0)
        high = spec.get("high", 1.0)
        if not isinstance(low, (list, tuple)) and not isinstance(high, (list, tuple)):
            return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        return gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), dtype=np.float32
        )
    if kind in {"discrete", "categorical"}:
        return gym.spaces.Discrete(int(spec.get("n", spec.get("size"))))
    if kind == "multidiscrete":
        return gym.spaces.MultiDiscrete(np.asarray(spec["nvec"], dtype=np.int64))
    if kind == "dict":
        return gym.spaces.Dict({key: _space_from_spec(value, observation=observation) for key, value in spec["spaces"].items()})
    raise ValueError(f"unsupported space declaration: {kind!r}")
