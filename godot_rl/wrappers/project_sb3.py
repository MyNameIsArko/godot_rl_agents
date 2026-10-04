from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from godot_rl.core.project_env import GodotProjectEnv


class GodotProjectVecEnv(VecEnv):
    """Adapt one Godot Gymnasium environment to Stable Baselines3."""

    def __init__(self, env: GodotProjectEnv) -> None:
        self.env = env
        self._raw_action_key, action_space = _extract_action_head(env.action_space)
        self._waiting = False
        super().__init__(1, env.observation_space, action_space)

    def reset(self) -> dict[str, np.ndarray] | np.ndarray:
        seed = self._seeds[0]
        options = self._options[0]
        if options:
            observation, info = self.env.reset(seed=seed, options=options)
        else:
            observation, info = self.env.reset(seed=seed)
        self.reset_infos[0] = info
        self._reset_seeds()
        self._reset_options()
        return _batch_observation(observation, self.observation_space)

    def step_async(self, actions: np.ndarray) -> None:
        if self._waiting:
            raise RuntimeError("step_async() called while a step is pending")
        try:
            if isinstance(actions, dict) or np.isscalar(actions):
                action = actions
            elif len(actions) == 1:
                action = actions[0]
            else:
                raise ValueError("GodotProjectVecEnv expects exactly one action")
            if self._raw_action_key is not None:
                action = {self._raw_action_key: action}
            self.env.step_send(action)
        except TypeError as exc:
            raise ValueError("GodotProjectVecEnv expects one batched action") from exc
        self._waiting = True

    def step_wait(self) -> tuple[dict[str, np.ndarray] | np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if not self._waiting:
            raise RuntimeError("step_wait() called without a pending step")
        try:
            observation, reward, terminated, truncated, info = self.env.step_recv()
        finally:
            self._waiting = False

        done = terminated or truncated
        info = dict(info)
        if done:
            info.setdefault("terminal_observation", observation)
            info["TimeLimit.truncated"] = truncated and not terminated
            observation, reset_info = self._reset_after_episode()
            self.reset_infos[0] = reset_info
        return (
            _batch_observation(observation, self.observation_space),
            np.asarray([reward], dtype=np.float32),
            np.asarray([done], dtype=bool),
            [info],
        )

    def _reset_after_episode(self) -> tuple[Any, dict[str, Any]]:
        seed = self._seeds[0]
        options = self._options[0]
        if options:
            result = self.env.reset(seed=seed, options=options)
        else:
            result = self.env.reset(seed=seed)
        self._reset_seeds()
        self._reset_options()
        return result

    def close(self) -> None:
        self.env.close()

    def get_images(self) -> list[np.ndarray | None]:
        return [None]

    def get_attr(self, attr_name: str, indices: Iterable[int] | int | None = None) -> list[Any]:
        return [getattr(self.env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Iterable[int] | int | None = None) -> None:
        for _ in self._get_indices(indices):
            setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Iterable[int] | int | None = None, **method_kwargs: Any) -> list[Any]:
        return [getattr(self.env, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Iterable[int] | int | None = None) -> list[bool]:
        return [isinstance(self.env, wrapper_class) for _ in self._get_indices(indices)]


def _batch_observation(observation: Any, space: gym.Space) -> dict[str, np.ndarray] | np.ndarray:
    if isinstance(space, gym.spaces.Dict):
        return {key: np.expand_dims(np.asarray(observation[key]), axis=0) for key in space.spaces}
    return np.expand_dims(np.asarray(observation), axis=0)


def _extract_action_head(space: gym.Space) -> tuple[str | None, gym.Space]:
    """Return the raw key and direct action space for one supported action head."""
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        return None, space
    if isinstance(space, gym.spaces.Dict) and len(space.spaces) == 1:
        key, child = next(iter(space.spaces.items()))
        if isinstance(child, (gym.spaces.Discrete, gym.spaces.Box)):
            return key, child
    raise ValueError("Stable Baselines3 supports one direct Discrete or Box action head")
