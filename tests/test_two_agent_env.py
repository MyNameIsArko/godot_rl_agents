import socket
import threading

import numpy as np
import pytest

from godot_rl.core.multi_agent_env import GodotMultiAgentEnv
from godot_rl.core.project_env import _GodotProcessSession
from godot_rl.core.protocol import ProtocolError, recv_frame, send_frame

OBS = {"obs": {"space": "box", "size": [2], "low": -1.0, "high": 1.0}}
ACTION = {"action": {"action_type": "discrete", "size": 2}}
BOX_ACTION = {"space": "box", "size": [1], "low": -1.0, "high": 1.0}


def _env_info(*, action0=ACTION, action1=ACTION, minor=0):
    return {
        "type": "env_info",
        "protocol": {"major": 2, "minor": minor},
        "agent_count": 2,
        "agents": [
            {"id": "player_0", "observation_space": OBS, "action_space": action0},
            {"id": "player_1", "observation_space": OBS, "action_space": action1},
        ],
    }


class FakePeer:
    def __init__(self, *, env_info=None, reset=None, step=None):
        self.server, self.client = socket.socketpair()
        self.server.settimeout(5)
        self.client.settimeout(5)
        self.env_info = env_info or _env_info()
        self.reset_response = reset or {"type": "reset", "agents": _records("reset")}
        self.step_response = step or {"type": "step", "agents": _records("step")}
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
                    send_frame(self.client, self.env_info)
                elif message["type"] == "reset":
                    send_frame(self.client, self.reset_response)
                elif message["type"] == "step":
                    send_frame(self.client, self.step_response)
                elif message["type"] == "close":
                    return
        except (OSError, ProtocolError):
            return

    def close(self):
        self.server.close()
        self.client.close()
        self.thread.join(timeout=2)


def _records(kind):
    if kind == "reset":
        return {agent_id: {"observation": {"obs": [0.0, 0.0]}, "info": {}} for agent_id in ("player_0", "player_1")}
    return {
        "player_0": {
            "observation": {"obs": [0.0, 0.0]},
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "info": {"terminal_observation": {"obs": [0.0, 0.0]}, "outcome": "win"},
        },
        "player_1": {
            "observation": {"obs": [0.0, 0.0]},
            "reward": -1.0,
            "terminated": True,
            "truncated": False,
            "info": {"terminal_observation": {"obs": [0.0, 0.0]}, "outcome": "loss"},
        },
    }


def make_env(peer):
    env = object.__new__(GodotMultiAgentEnv)
    env._session = object.__new__(_GodotProcessSession)
    env._session.connection = peer.server
    env._session.connection.settimeout(5)
    env._session.process = None
    env._session._listener = None
    env._session.protocol_major = 2
    env._session._closed = False
    env._closed = False
    env._handshake()
    env._read_env_info()
    return env


def test_env_info_reset_and_step_use_both_agents():
    peer = FakePeer()
    env = make_env(peer)
    observations, infos = env.reset(seed=17)
    assert list(env.observation_spaces) == ["player_0", "player_1"]
    assert list(observations) == ["player_0", "player_1"]
    assert infos == {"player_0": {}, "player_1": {}}
    observations, rewards, terminated, truncated, step_infos = env.step(
        {"player_0": {"action": 0}, "player_1": {"action": 1}}
    )
    assert set(observations) == {"player_0", "player_1"}
    np.testing.assert_array_equal(observations["player_0"]["obs"], [0.0, 0.0])
    np.testing.assert_array_equal(observations["player_1"]["obs"], [0.0, 0.0])
    assert rewards == {"player_0": 1.0, "player_1": -1.0}
    assert terminated == {"player_0": True, "player_1": True}
    assert truncated == {"player_0": False, "player_1": False}
    assert step_infos == {
        "player_0": {
            "terminal_observation": {"obs": [0.0, 0.0]},
            "outcome": "win",
        },
        "player_1": {
            "terminal_observation": {"obs": [0.0, 0.0]},
            "outcome": "loss",
        },
    }
    assert peer.messages[-1] == {
        "type": "step",
        "actions": {"player_0": {"action": 0}, "player_1": {"action": 1}},
    }
    env.close()
    env.close()
    peer.close()


@pytest.mark.parametrize("agent_ids", [["player_0", "player_0"], ["player_0"], ["player_0", "other"]])
def test_env_info_rejects_invalid_identifiers(agent_ids):
    response = {
        "type": "env_info",
        "protocol": {"major": 2, "minor": 0},
        "agent_count": 2,
        "agents": [{"id": agent_id, "observation_space": OBS, "action_space": ACTION} for agent_id in agent_ids],
    }
    peer = FakePeer(env_info=response)
    with pytest.raises(ValueError, match="env_info.agents"):
        make_env(peer)
    peer.close()


def test_protocol_v2_requires_minor_zero():
    peer = FakePeer(env_info=_env_info(minor=1))
    with pytest.raises(ProtocolError, match=r"env_info\.protocol\.minor must be 0"):
        make_env(peer)
    peer.close()


def test_action_spaces_must_match():
    peer = FakePeer(env_info=_env_info(action1=BOX_ACTION))
    with pytest.raises(ProtocolError, match="action_space"):
        make_env(peer)
    peer.close()


@pytest.mark.parametrize(
    "action_space",
    [
        {"space": "multidiscrete", "nvec": [2, 2]},
        {
            "type": "dict",
            "spaces": {
                "left": {"action_type": "discrete", "size": 2},
                "right": {"action_type": "discrete", "size": 2},
            },
        },
        {
            "type": "dict",
            "spaces": {
                "action": {
                    "type": "dict",
                    "spaces": {"nested": {"action_type": "discrete", "size": 2}},
                }
            },
        },
    ],
)
def test_action_space_rejects_unsupported_shapes(action_space):
    peer = FakePeer(env_info=_env_info(action0=action_space, action1=action_space))
    with pytest.raises(ProtocolError, match="Discrete or Box"):
        make_env(peer)
    peer.close()


@pytest.mark.parametrize(
    "agent_map",
    [
        {"player_0": _records("reset")["player_0"]},
        {
            "player_0": _records("reset")["player_0"],
            "player_1": _records("reset")["player_1"],
            "spectator": _records("reset")["player_0"],
        },
    ],
)
def test_reset_rejects_missing_or_unknown_agent_ids(agent_map):
    peer = FakePeer(reset={"type": "reset", "agents": agent_map})
    env = make_env(peer)
    with pytest.raises(ProtocolError, match=r"reset\.agents\.(player_1|spectator)"):
        env.reset()
    env.close()
    peer.close()


@pytest.mark.parametrize(
    "agent_map",
    [
        {"player_0": _records("step")["player_0"]},
        {
            "player_0": _records("step")["player_0"],
            "player_1": _records("step")["player_1"],
            "spectator": _records("step")["player_0"],
        },
    ],
)
def test_step_response_rejects_missing_or_unknown_agent_ids(agent_map):
    peer = FakePeer(step={"type": "step", "agents": agent_map})
    env = make_env(peer)
    env.reset()
    with pytest.raises(ProtocolError, match=r"step\.agents\.(player_1|spectator)"):
        env.step({"player_0": {"action": 0}, "player_1": {"action": 1}})
    env.close()
    peer.close()


@pytest.mark.parametrize(
    "actions",
    [
        {"player_0": {"action": 0}},
        {"player_0": {"action": 0}, "player_1": {"action": 1}, "spectator": {"action": 0}},
    ],
)
def test_step_actions_reject_missing_or_unknown_agent_ids_before_transmission(actions):
    peer = FakePeer()
    env = make_env(peer)
    with pytest.raises(ProtocolError, match=r"step\.actions\.(player_1|spectator)"):
        env.step(actions)
    assert not any(message["type"] == "step" for message in peer.messages)
    env.close()
    peer.close()


def test_invalid_action_is_rejected_before_transmission():
    peer = FakePeer()
    env = make_env(peer)
    with pytest.raises(ValueError, match="step.actions.player_1"):
        env.step({"player_0": {"action": 0}, "player_1": {"action": 2}})
    assert not any(message["type"] == "step" for message in peer.messages)
    env.close()
    peer.close()


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_nonfinite_reward_reports_agent_field_path(reward):
    peer = FakePeer()
    env = make_env(peer)
    env.reset()
    response = {"type": "step", "agents": _records("step")}
    response["agents"]["player_1"]["reward"] = reward
    env._session.receive = lambda: response
    with pytest.raises(ProtocolError, match=r"step\.agents\.player_1\.reward"):
        env.step({"player_0": {"action": 0}, "player_1": {"action": 1}})
    env.close()
    peer.close()


def test_terminal_outcomes_and_flags_are_validated():
    peer = FakePeer()
    env = make_env(peer)
    env.reset()
    result = env.step({"player_0": {"action": 0}, "player_1": {"action": 1}})
    assert result[2] == {"player_0": True, "player_1": True}
    assert result[3] == {"player_0": False, "player_1": False}
    env.close()
    peer.close()


def _assert_invalid_step(step_agents, match):
    peer = FakePeer(step={"type": "step", "agents": step_agents})
    env = make_env(peer)
    env.reset()
    with pytest.raises(ProtocolError, match=match):
        env.step({"player_0": {"action": 0}, "player_1": {"action": 1}})
    env.close()
    peer.close()


def test_step_rejects_unsynchronized_termination():
    agents = _records("step")
    agents["player_1"]["terminated"] = False
    _assert_invalid_step(agents, r"terminated.*must match")


def test_step_rejects_unsynchronized_truncation():
    agents = _records("step")
    agents["player_0"]["terminated"] = False
    agents["player_1"]["terminated"] = False
    agents["player_1"]["truncated"] = True
    _assert_invalid_step(agents, r"truncated.*must match")


def test_step_rejects_termination_and_truncation_together():
    agents = _records("step")
    agents["player_0"]["truncated"] = True
    agents["player_1"]["truncated"] = True
    _assert_invalid_step(agents, r"cannot both be true")


def test_terminal_step_requires_terminal_observation_for_each_agent():
    agents = _records("step")
    del agents["player_1"]["info"]["terminal_observation"]
    _assert_invalid_step(agents, r"step\.agents\.player_1\.info\.terminal_observation")


def test_terminal_step_requires_complementary_outcomes():
    agents = _records("step")
    agents["player_1"]["info"]["outcome"] = "win"
    _assert_invalid_step(agents, "outcomes must be complementary")


def test_invalid_observation_names_agent_field():
    response = {"type": "reset", "agents": _records("reset")}
    response["agents"]["player_1"]["observation"] = {"obs": [2.0, 0.0]}
    peer = FakePeer(reset=response)
    env = make_env(peer)
    with pytest.raises(ValueError, match=r"reset\.agents\.player_1\.observation\.obs"):
        env.reset()
    env.close()
    peer.close()


def test_spaces_must_match():
    different = {"obs": {"space": "box", "size": [3], "low": -1.0, "high": 1.0}}
    response = {
        "type": "env_info",
        "protocol": {"major": 2, "minor": 0},
        "agent_count": 2,
        "agents": [
            {"id": "player_0", "observation_space": OBS, "action_space": ACTION},
            {"id": "player_1", "observation_space": different, "action_space": ACTION},
        ],
    }
    peer = FakePeer(env_info=response)
    with pytest.raises(ValueError, match="observation_space"):
        make_env(peer)
    peer.close()
