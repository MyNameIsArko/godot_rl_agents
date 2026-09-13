import numpy as np

from godot_rl.core.godot_env import GodotEnv


def make_env(response):
    env = object.__new__(GodotEnv)
    env._get_json_dict = lambda: response
    env._process_obs = lambda observation: observation
    return env


def test_step_recv_preserves_terminated_and_truncated_separately():
    env = make_env(
        {
            "obs": [{"obs": np.zeros(1)}],
            "reward": [1.0],
            "terminated": [True],
            "truncated": [False],
        }
    )

    _, _, terminated, truncated, _ = env.step_recv()

    assert terminated == [True]
    assert truncated == [False]


def test_step_recv_supports_legacy_done_response():
    env = make_env(
        {
            "obs": [{"obs": np.zeros(1)}],
            "reward": [1.0],
            "done": [True],
        }
    )

    _, _, terminated, truncated, _ = env.step_recv()

    assert terminated == [True]
    assert truncated == [False]
