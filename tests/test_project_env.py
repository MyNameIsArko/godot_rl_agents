import socket
import subprocess
import sys
import threading

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import godot_rl.core.project_env as env_module
from godot_rl.core.project_env import (
    GodotProjectEnv,
    _GodotProcessSession,
    _space_from_agent_spec,
)
from godot_rl.core.protocol import ProtocolError, recv_frame, send_frame

OBSERVATION_SPACE = {"obs": {"space": "box", "size": [2], "low": -1.0, "high": 1.0}}
ACTION_SPACE = {"action": {"action_type": "discrete", "size": 2}}


class FakePeer:
    def __init__(self, *, invalid_observation=False, protocol=None, terminated=True, truncated=False):
        self.server, self.client = socket.socketpair()
        self.server.settimeout(5)
        self.client.settimeout(5)
        self.invalid_observation = invalid_observation
        self.protocol = protocol or {"major": 1, "minor": 0}
        self.terminated = terminated
        self.truncated = truncated
        self.messages = []
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        try:
            while True:
                message = recv_frame(self.client)
                self.messages.append(message)
                if message["type"] == "handshake":
                    send_frame(self.client, message)
                elif message["type"] == "env_info":
                    send_frame(
                        self.client,
                        {
                            "type": "env_info",
                            "protocol": self.protocol,
                            "observation_space": OBSERVATION_SPACE,
                            "action_space": ACTION_SPACE,
                            "agent_count": 1,
                        },
                    )
                elif message["type"] == "reset":
                    send_frame(self.client, {"type": "reset", "observation": self.observation(), "info": {}})
                elif message["type"] == "step":
                    send_frame(
                        self.client,
                        {
                            "type": "step",
                            "observation": self.observation(),
                            "reward": 1.0,
                            "terminated": self.terminated,
                            "truncated": self.truncated,
                            "info": {"source": "fake"},
                        },
                    )
                elif message["type"] == "close":
                    return
        except (OSError, EOFError):
            return

    def observation(self):
        return {"obs": [2.0, 0.0]} if self.invalid_observation else {"obs": [0.0, 0.0]}

    def close(self):
        self.server.close()
        self.client.close()
        self.thread.join(timeout=2)


def make_env(peer):
    env = object.__new__(GodotProjectEnv)
    env._session = make_session(peer)
    env._closed = False
    env.observation_space = _space_from_agent_spec(OBSERVATION_SPACE, observation=True)
    env.action_space = _space_from_agent_spec(ACTION_SPACE, observation=False)
    env._handshake()
    env._read_env_info()
    return env


def make_session(peer):
    session = object.__new__(_GodotProcessSession)
    session.connection = peer.server
    session.connection.settimeout(5)
    session.process = None
    session._listener = None
    session.protocol_major = 1
    session._closed = False
    return session


def test_handshake_spaces_match_and_reset_returns_observation_and_info():
    peer = FakePeer()
    env = make_env(peer)
    observation, info = env.reset(seed=11)
    assert env.observation_space == _space_from_agent_spec(OBSERVATION_SPACE, observation=True)
    assert env.action_space == _space_from_agent_spec(ACTION_SPACE, observation=False)
    assert np.array_equal(observation["obs"], np.zeros(2, dtype=np.float32))
    assert info == {}
    assert peer.messages[-1] == {"type": "reset", "seed": 11}
    env.close()
    peer.close()


def test_step_returns_five_values_with_separate_terminal_flags():
    peer = FakePeer()
    env = make_env(peer)
    env.reset()
    result = env.step({"action": 1})
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert observation["obs"].shape == (2,)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info == {"source": "fake"}
    assert peer.messages[-1] == {"type": "step", "action": {"action": 1}}
    env.close()
    peer.close()


def test_step_preserves_truncation_without_termination():
    peer = FakePeer(terminated=False, truncated=True)
    env = make_env(peer)
    env.reset()
    _, _, terminated, truncated, _ = env.step({"action": 0})
    assert terminated is False
    assert truncated is True
    env.close()
    peer.close()


def test_invalid_observation_error_names_the_field():
    peer = FakePeer(invalid_observation=True)
    env = make_env(peer)
    with pytest.raises(ValueError, match=r"observation\.obs"):
        env.reset()
    env.close()
    peer.close()


def test_close_is_idempotent():
    peer = FakePeer()
    env = make_env(peer)
    env.close()
    env.close()
    peer.close()


@pytest.mark.parametrize("protocol", ["bad", {"major": 2, "minor": 0}])
def test_env_info_protocol_is_required_and_major_compatible(protocol):
    peer = FakePeer(protocol=protocol)
    env = object.__new__(GodotProjectEnv)
    env._session = make_session(peer)
    env._closed = False
    env._handshake()
    with pytest.raises(ProtocolError, match="protocol"):
        env._read_env_info()
    env.close()
    peer.close()


def test_gymnasium_checker_passes_against_fake_peer():
    peer = FakePeer()
    env = make_env(peer)
    check_env(env, skip_render_check=True)
    env.close()
    peer.close()


class FakePopen:
    def __init__(self):
        self.command = None
        self.kwargs = None

    def __call__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        return object()


def make_launch_env(show_window=False):
    session = object.__new__(_GodotProcessSession)
    session.godot_path = "/godot"
    session.project_path = "/project"
    session.scene = "res://train.tscn"
    session.port = 11008
    session.seed_value = 7
    session.speedup = 8.0
    session.show_window = show_window
    return session


@pytest.mark.parametrize(
    ("os_name", "show_window"),
    [("posix", False), ("nt", False), ("nt", True)],
)
def test_launch_command_and_process_flags(monkeypatch, os_name, show_window):
    popen = FakePopen()
    monkeypatch.setattr(env_module.subprocess, "Popen", popen)
    monkeypatch.setattr(env_module.os, "name", os_name)
    env = make_launch_env(show_window)

    env._launch_godot()

    assert popen.command == [
        "/godot",
        "--path",
        "/project",
        *( [] if show_window else ["--headless", "--disable-render-loop"]),
        "--scene",
        "res://train.tscn",
        "--port=11008",
        "--env_seed=7",
        "--speedup=8.0",
    ]
    assert popen.kwargs["shell"] is False
    if os_name == "posix":
        assert popen.kwargs["start_new_session"] is True
    else:
        process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        assert popen.kwargs["creationflags"] & process_group
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        assert bool(popen.kwargs["creationflags"] & no_window) is not show_window


class EscalatingProcess:
    def __init__(self):
        self.calls = []

    def poll(self):
        self.calls.append("poll")

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        raise subprocess.TimeoutExpired("godot", timeout)

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


def test_close_escalates_with_bounded_waits(monkeypatch):
    process = EscalatingProcess()
    session = object.__new__(_GodotProcessSession)
    session.connection = None
    session._listener = None
    session.process = process
    session._closed = False
    monkeypatch.setattr(env_module.os, "name", "nt")

    session.close()

    assert [call for call in process.calls if call != "poll"] == [
        ("wait", 2.0),
        "terminate",
        ("wait", 2.0),
        "kill",
        ("wait", 2.0),
    ]


def test_close_terminates_a_real_process_and_can_be_called_twice():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    session = object.__new__(_GodotProcessSession)
    session.connection = None
    session._listener = None
    session.process = process
    session._closed = False
    try:
        session.close()
        session.close()
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
    assert process.returncode is not None
