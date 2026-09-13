import gymnasium as gym
import numpy as np

from godot_rl.core.godot_gym_env import GodotGymEnv


class FakeGodotEnv:
    num_envs = 1
    observation_spaces = [gym.spaces.Dict({"obs": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})]
    tuple_action_spaces = [gym.spaces.Tuple((gym.spaces.Discrete(2),))]

    def __init__(self, *args, **kwargs):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))
        return [{"obs": np.zeros(2, dtype=np.float32)}], [{}]

    def step(self, action, order_ij=False):
        self.calls.append(("step", action, order_ij))
        return [{"obs": np.zeros(2, dtype=np.float32)}], [1.0], [True], [False], [{}]

    def close(self):
        self.calls.append(("close",))


def test_single_action_is_packed_for_existing_godot_env(monkeypatch):
    fake = FakeGodotEnv()
    monkeypatch.setattr("godot_rl.core.godot_gym_env.GodotEnv", lambda **kwargs: fake)
    env = GodotGymEnv()

    observation, info = env.reset(seed=4)
    result = env.step(1)

    assert observation["obs"].shape == (2,)
    assert info == {}
    assert result[1:] == (1.0, True, False, {})
    assert fake.calls[0] == ("reset", {"seed": 4, "options": None})
    assert fake.calls[1] == ("step", [[1]], True)


def test_invalid_action_is_rejected(monkeypatch):
    fake = FakeGodotEnv()
    monkeypatch.setattr("godot_rl.core.godot_gym_env.GodotEnv", lambda **kwargs: fake)
    env = GodotGymEnv()

    try:
        env.step(2)
    except ValueError as error:
        assert "action" in str(error)
    else:
        raise AssertionError("an invalid action was accepted")
