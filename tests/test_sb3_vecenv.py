import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


class FakeEnv:
    num_envs = 1
    observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.calls = []

    def step_send(self, action):
        self.calls.append(("send", action))

    def step_recv(self):
        self.calls.append(("recv", None))
        return [{"obs": np.array([0.5, 0.25])}], [1.0], [True], [False], [{}]

    def reset(self, seed=None):
        self.calls.append(("reset", seed))
        return [{"obs": np.zeros(2)}], [{}]

    def close(self):
        self.calls.append(("close", None))


def make_vec():
    vec = object.__new__(StableBaselinesGodotEnv)
    vec.envs = [FakeEnv()]
    vec.n_parallel = 1
    vec._waiting = False
    VecEnv.__init__(vec, 1, vec.envs[0].observation_space, vec.envs[0].action_space)
    return vec


def test_step_async_sends_before_step_wait_receives():
    vec = make_vec()

    vec.step_async(np.array([1]))
    assert len(vec.envs[0].calls) == 1
    assert vec.envs[0].calls[0][0] == "send"
    assert np.array_equal(vec.envs[0].calls[0][1], np.array([1]))
    observation, rewards, dones, infos = vec.step_wait()

    assert vec.envs[0].calls[-1] == ("recv", None)
    assert observation["obs"].tolist() == [[0.5, 0.25]]
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert infos[0]["terminal_observation"]["obs"].tolist() == [0.5, 0.25]


def test_pending_steps_are_rejected():
    vec = make_vec()
    vec.step_async(np.array([0]))

    try:
        vec.step_async(np.array([1]))
    except RuntimeError as error:
        assert "pending" in str(error)
    else:
        raise AssertionError("a second pending step was accepted")
