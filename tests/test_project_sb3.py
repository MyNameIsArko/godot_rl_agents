import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from godot_rl.wrappers.project_sb3 import GodotProjectVecEnv, _extract_action_head


class FakeEnv:
    observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)})
    action_space = gym.spaces.Dict({"action": gym.spaces.Discrete(2)})
    render_mode = None

    def __init__(self, *, terminated=False, truncated=False):
        self.terminated = terminated
        self.truncated = truncated
        self.calls = []
        self.reset_seeds = []

    def reset(self, *, seed=None, options=None):
        self.calls.append("reset")
        self.reset_seeds.append(seed)
        return {"obs": np.array([0.0, 0.0], dtype=np.float32)}, {"reset": True}

    def step_send(self, action):
        self.calls.append("send")
        self.action = action

    def step_recv(self):
        self.calls.append("recv")
        return (
            {"obs": np.array([0.5, 0.25], dtype=np.float32)},
            1.0,
            self.terminated,
            self.truncated,
            {"source": "fake"},
        )

    def close(self):
        self.calls.append("close")

    def render(self):
        return None


def make_vec(**kwargs):
    raw = FakeEnv(**kwargs)
    return GodotProjectVecEnv(raw), raw


def test_step_async_sends_without_receiving_and_step_wait_receives_once():
    vec, raw = make_vec()
    vec.step_async(np.array([1]))
    assert raw.calls == ["send"]
    assert raw.action == {"action": 1}
    vec.step_wait()
    assert raw.calls == ["send", "recv"]


def test_async_call_order_is_enforced():
    vec, _ = make_vec()
    with pytest.raises(RuntimeError, match="pending"):
        vec.step_wait()
    vec.step_async(np.array([0]))
    with pytest.raises(RuntimeError, match="pending"):
        vec.step_async(np.array([1]))
    vec.step_wait()


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_terminal_observation_truncation_and_reset(terminated, truncated):
    vec, raw = make_vec(terminated=terminated, truncated=truncated)
    observation, rewards, dones, infos = vec.step(np.array([0]))
    assert observation["obs"].tolist() == [[0.0, 0.0]]
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert infos[0]["terminal_observation"]["obs"].tolist() == [0.5, 0.25]
    assert infos[0]["TimeLimit.truncated"] is (truncated and not terminated)
    assert raw.calls == ["send", "recv", "reset"]


def test_seed_is_used_by_the_next_reset_only():
    vec, raw = make_vec()
    assert vec.seed(42) == [42]
    vec.reset()
    vec.reset()
    assert raw.reset_seeds == [42, None]


def test_vecenv_methods_delegate_to_the_single_raw_environment():
    vec, raw = make_vec()
    assert vec.get_attr("render_mode") == [None]
    assert vec.env_method("render") == [None]
    assert vec.env_is_wrapped(gym.Wrapper) == [False]
    vec.set_attr("marker", "value")
    assert raw.marker == "value"
    vec.close()
    assert raw.calls == ["close"]


@pytest.mark.parametrize(
    ("space", "key", "expected"),
    [
        (gym.spaces.Discrete(2), None, gym.spaces.Discrete(2)),
        (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), None, gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)),
        (gym.spaces.Dict({"action": gym.spaces.Discrete(2)}), "action", gym.spaces.Discrete(2)),
        (gym.spaces.Dict({"action": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)}), "action", gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)),
    ],
)
def test_extract_action_head_supports_one_direct_or_dict_head(space, key, expected):
    actual_key, actual_space = _extract_action_head(space)
    assert actual_key == key
    assert actual_space == expected


@pytest.mark.parametrize(
    "space",
    [
        gym.spaces.Dict({}),
        gym.spaces.Dict({"left": gym.spaces.Discrete(2), "right": gym.spaces.Discrete(2)}),
        gym.spaces.Dict({"action": gym.spaces.Dict({"nested": gym.spaces.Discrete(2)})}),
        gym.spaces.MultiDiscrete([2, 2]),
        gym.spaces.Tuple((gym.spaces.Discrete(2),)),
    ],
)
def test_extract_action_head_rejects_zero_or_multiple_heads(space):
    with pytest.raises(ValueError, match="one direct Discrete or Box"):
        _extract_action_head(space)
