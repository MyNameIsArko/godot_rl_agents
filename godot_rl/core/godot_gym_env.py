"""Gymnasium interface for a single-agent Godot environment."""

from typing import Any, Optional

import gymnasium as gym

from godot_rl.core.godot_env import GodotEnv


class GodotGymEnv(gym.Env):
    """Adapt the existing Godot environment to the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, env_path: Optional[str] = None, **kwargs: Any) -> None:
        kwargs["convert_action_space"] = False
        self.env = GodotEnv(env_path=env_path, **kwargs)
        if self.env.num_envs != 1:
            self.env.close()
            raise ValueError("GodotGymEnv supports exactly one agent")

        self.observation_space = self.env.observation_spaces[0]
        action_heads = self.env.tuple_action_spaces[0].spaces
        self._single_action = len(action_heads) == 1
        self.action_space = action_heads[0] if self._single_action else self.env.tuple_action_spaces[0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations[0], infos[0]

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action is outside the declared action space")
        action_heads = [action] if self._single_action else action
        observations, rewards, terminated, truncated, infos = self.env.step(
            [action_heads], order_ij=True
        )
        return observations[0], rewards[0], terminated[0], truncated[0], infos[0]

    def render(self):
        return None

    def close(self):
        self.env.close()
