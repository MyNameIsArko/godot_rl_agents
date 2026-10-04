from __future__ import annotations

import math
import numbers
import pathlib
from typing import Any

import gymnasium as gym
import numpy as np

from godot_rl.core.project_env import (
    _GodotProcessSession,
    _json_value,
    _space_from_agent_spec,
)
from godot_rl.core.protocol import (
    PROTOCOL_MINOR,
    PROTOCOL_V2_MAJOR,
    ProtocolError,
    validate_finite,
)

AGENT_IDS = ("player_0", "player_1")


class GodotMultiAgentEnv:
    """Two-agent shared-world environment backed by one Godot process."""

    agent_ids = AGENT_IDS

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
                protocol_major=PROTOCOL_V2_MAJOR,
            )
            self.process = self._session.process
            self._listener = self._session._listener
            self.connection = self._session.connection
            self._read_env_info()
        except BaseException:
            self.close()
            raise

    def _handshake(self) -> None:
        self._session.handshake()

    def _read_env_info(self) -> None:
        self._session.send(
            {"type": "env_info", "protocol": {"major": PROTOCOL_V2_MAJOR, "minor": PROTOCOL_MINOR}}
        )
        response = self._session.receive()
        _require_type(response, "env_info", "env_info")
        _require_protocol(response, "env_info")
        if response["protocol"]["major"] != PROTOCOL_V2_MAJOR:
            raise ProtocolError(
                "env_info.protocol.major mismatch: "
                f"peer={response['protocol']['major']}, expected={PROTOCOL_V2_MAJOR}"
            )
        if response["protocol"].get("minor") != PROTOCOL_MINOR:
            raise ProtocolError("env_info.protocol.minor must be 0")
        if response.get("agent_count") != len(AGENT_IDS):
            raise ValueError("env_info.agent_count must be 2")
        agents = response.get("agents")
        if not isinstance(agents, list) or len(agents) != len(AGENT_IDS):
            raise ValueError("env_info.agents must be a two-item list")
        if any(not isinstance(agent, dict) for agent in agents):
            index = next(index for index, agent in enumerate(agents) if not isinstance(agent, dict))
            raise ValueError(f"env_info.agents[{index}] must be a dictionary")
        ids = [agent.get("id") for agent in agents]
        if ids != list(AGENT_IDS):
            raise ValueError("env_info.agents.id must be exactly ['player_0', 'player_1'] in order")
        observation_spaces: dict[str, gym.Space] = {}
        action_spaces: dict[str, gym.Space] = {}
        for index, agent in enumerate(agents):
            field = f"env_info.agents[{index}]"
            _require_keys(agent, {"id", "observation_space", "action_space"}, field)
            observation_spaces[agent["id"]] = _parse_space(
                agent["observation_space"], f"{field}.observation_space", True
            )
            action_spaces[agent["id"]] = _parse_space(
                agent["action_space"], f"{field}.action_space", False
            )
        if observation_spaces[AGENT_IDS[0]] != observation_spaces[AGENT_IDS[1]]:
            raise ProtocolError("env_info.agents observation_space values must match")
        if action_spaces[AGENT_IDS[0]] != action_spaces[AGENT_IDS[1]]:
            raise ProtocolError("env_info.agents action_space values must match")
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
        self._require_open()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        message: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            message["seed"] = seed
        self._session.send(message)
        response = self._session.receive()
        _require_type(response, "reset", "reset")
        records = _agent_records(response.get("agents"), "reset.agents")
        observations: dict[str, object] = {}
        infos: dict[str, dict[str, object]] = {}
        for agent_id in AGENT_IDS:
            field = f"reset.agents.{agent_id}"
            record = records[agent_id]
            _require_keys(record, {"observation", "info"}, field)
            observations[agent_id] = _coerce_value(
                record["observation"], self.observation_spaces[agent_id], f"{field}.observation"
            )
            infos[agent_id] = _coerce_info(record["info"], f"{field}.info")
        return observations, infos

    def step(
        self, actions: dict[str, object]
    ) -> tuple[
        dict[str, object],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, object]],
    ]:
        self._require_open()
        actions = _require_agent_ids(actions, "step.actions")
        encoded_actions: dict[str, object] = {}
        for agent_id in AGENT_IDS:
            value = _coerce_value(
                actions[agent_id], self.action_spaces[agent_id], f"step.actions.{agent_id}"
            )
            encoded_actions[agent_id] = _json_value(value)
        self._session.send({"type": "step", "actions": encoded_actions})

        response = self._session.receive()
        _require_type(response, "step", "step")
        if "done" in response:
            raise ProtocolError("protocol version 2 step response must not contain done")
        records = _agent_records(response.get("agents"), "step.agents")
        observations: dict[str, object] = {}
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        infos: dict[str, dict[str, object]] = {}
        outcomes: dict[str, str] = {}
        for agent_id in AGENT_IDS:
            field = f"step.agents.{agent_id}"
            record = records[agent_id]
            _require_keys(
                record, {"observation", "reward", "terminated", "truncated", "info"}, field
            )
            observations[agent_id] = _coerce_value(
                record["observation"], self.observation_spaces[agent_id], f"{field}.observation"
            )
            reward = record["reward"]
            if (
                isinstance(reward, bool)
                or not isinstance(reward, numbers.Real)
                or not math.isfinite(reward)
            ):
                raise ProtocolError(f"{field}.reward must be a finite number")
            rewards[agent_id] = float(reward)
            for flag in ("terminated", "truncated"):
                value = record[flag]
                if not isinstance(value, bool):
                    raise ProtocolError(f"{field}.{flag} must be a boolean")
            terminated[agent_id] = record["terminated"]
            truncated[agent_id] = record["truncated"]
            if terminated[agent_id] and truncated[agent_id]:
                raise ProtocolError(f"{field} cannot both be true")
            infos[agent_id] = _coerce_info(record["info"], f"{field}.info")
            if terminated[agent_id] or truncated[agent_id]:
                info = infos[agent_id]
                if "terminal_observation" not in info:
                    raise ProtocolError(f"{field}.info.terminal_observation is required")
                if "outcome" not in info:
                    raise ProtocolError(f"{field}.info.outcome is required")
                _coerce_value(
                    info["terminal_observation"],
                    self.observation_spaces[agent_id],
                    f"{field}.info.terminal_observation",
                )
                outcome = info["outcome"]
                if outcome not in {"win", "loss", "draw"}:
                    raise ProtocolError(f"{field}.info.outcome must be win, loss, or draw")
                outcomes[agent_id] = outcome

        if len(set(terminated.values())) != 1:
            raise ProtocolError("step.agents terminated flags must match")
        if len(set(truncated.values())) != 1:
            raise ProtocolError("step.agents truncated flags must match")
        if any(terminated.values()) and any(truncated.values()):
            raise ProtocolError("step.agents cannot be terminated and truncated together")
        if outcomes:
            if set(outcomes) != set(AGENT_IDS):
                raise ProtocolError("step.agents.info.outcome is required for both agents")
            if sorted(outcomes.values()) not in (["draw", "draw"], ["loss", "win"]):
                raise ProtocolError("step.agents outcomes must be complementary")
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GodotMultiAgentEnv is closed")


def _require_type(message: dict[str, Any], expected: str, field: str) -> None:
    if message.get("type") != expected:
        raise ProtocolError(f"{field}.type must be {expected!r}")


def _require_protocol(message: dict[str, Any], field: str) -> None:
    protocol = message.get("protocol")
    if not isinstance(protocol, dict):
        raise ProtocolError(f"{field}.protocol must be a dictionary")
    _require_keys(protocol, {"major", "minor"}, f"{field}.protocol")
    if type(protocol["major"]) is not int or protocol["major"] != PROTOCOL_V2_MAJOR:
        raise ProtocolError(f"{field}.protocol.major must be {PROTOCOL_V2_MAJOR}")
    if type(protocol["minor"]) is not int or protocol["minor"] != PROTOCOL_MINOR:
        raise ProtocolError(f"{field}.protocol.minor must be {PROTOCOL_MINOR}")


def _require_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    keys = set(value)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ProtocolError(f"{field}.{min(missing)} is required")
    if extra:
        raise ProtocolError(f"{field}.{min(extra)} is unknown")


def _agent_records(value: Any, field: str) -> dict[str, dict[str, Any]]:
    value = _require_agent_ids(value, field)
    records: dict[str, dict[str, Any]] = {}
    for agent_id in AGENT_IDS:
        if not isinstance(value[agent_id], dict):
            raise ProtocolError(f"{field}.{agent_id} must be a dictionary")
        records[agent_id] = value[agent_id]
    return records


def _require_agent_ids(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be a dictionary")
    missing = set(AGENT_IDS) - set(value)
    extra = set(value) - set(AGENT_IDS)
    if missing:
        raise ProtocolError(f"{field}.{min(missing)} is missing")
    if extra:
        raise ProtocolError(f"{field}.{min(extra)} is unknown")
    return value


def _parse_space(spec: Any, field: str, observation: bool) -> gym.Space:
    try:
        space = _space_from_agent_spec(spec, observation=observation)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {exc}") from exc
    if observation:
        if not isinstance(space, (gym.spaces.Box, gym.spaces.Dict)):
            raise ProtocolError(f"{field} must be a Box or Dict space")
        return space
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        return space
    if isinstance(space, gym.spaces.Dict) and len(space.spaces) == 1:
        child = next(iter(space.spaces.values()))
        if isinstance(child, (gym.spaces.Discrete, gym.spaces.Box)):
            return space
    raise ProtocolError(f"{field} must contain one direct Discrete or Box action head")


def _coerce_info(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be a dictionary")
    validate_finite(value, field)
    return value


def _coerce_value(value: Any, space: gym.Space, field: str) -> Any:
    validate_finite(value, field)
    if isinstance(space, gym.spaces.Dict):
        if not isinstance(value, dict):
            raise ProtocolError(f"{field} must be a dictionary")
        expected = set(space.spaces)
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ProtocolError(f"{field}.{min(missing)} is missing")
        if extra:
            raise ProtocolError(f"{field}.{min(extra)} is unknown")
        return {
            key: _coerce_value(value[key], child, f"{field}.{key}")
            for key, child in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Box):
        try:
            value = np.asarray(value, dtype=space.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}: {exc}") from exc
    if not space.contains(value):
        raise ValueError(f"{field} is outside the declared space")
    return value
