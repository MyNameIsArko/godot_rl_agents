from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import can_import, lod_to_dol


class StableBaselinesGodotEnv(VecEnv):
    def __init__(
        self,
        env_path: Optional[str] = None,
        n_parallel: int = 1,
        seed: int = 0,
        **kwargs,
    ) -> None:
        # If we are doing editor training, n_parallel must be 1
        if env_path is None and n_parallel > 1:
            raise ValueError("You must provide the path to a exported game executable if n_parallel > 1")

        # Define the default port
        port = kwargs.pop("port", GodotEnv.DEFAULT_PORT)

        # Create a list of GodotEnv instances
        self.envs = [
            GodotEnv(
                env_path=env_path,
                convert_action_space=True,
                port=port + p,
                seed=seed + p,
                **kwargs,
            )
            for p in range(n_parallel)
        ]

        # Store the number of parallel environments
        self.n_parallel = n_parallel

        # Check the action space for validity
        self._check_valid_action_space()
        super().__init__(
            self.envs[0].num_envs * n_parallel,
            self.envs[0].observation_space,
            self.envs[0].action_space,
        )
        self._waiting = False

    def _check_valid_action_space(self) -> None:
        # Check if the action space is a tuple space with multiple spaces
        action_space = self.envs[0].action_space
        if isinstance(action_space, gym.spaces.Tuple):
            assert (
                len(action_space.spaces) == 1
            ), f"sb3 supports a single action space, this env contains multiple spaces {action_space}"

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        self.step_async(action)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        if self._waiting:
            raise RuntimeError("step_async() called while a step is pending")

        num_envs = self.envs[0].num_envs
        for i in range(self.n_parallel):
            self.envs[i].step_send(actions[i * num_envs : (i + 1) * num_envs])
        self._waiting = True

    def step_wait(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if not self._waiting:
            raise RuntimeError("step_wait() called without a pending step")

        # Initialize lists for collecting results
        all_obs = []
        all_rewards = []
        all_term = []
        all_trunc = []
        all_info = []

        try:
            for i in range(self.n_parallel):
                obs, reward, term, trunc, info = self.envs[i].step_recv()
                all_obs.extend(obs)
                all_rewards.extend(reward)
                all_term.extend(term)
                all_trunc.extend(trunc)
                all_info.extend(info)
        finally:
            self._waiting = False

        # Convert list of dictionaries to dictionary of lists
        obs = lod_to_dol(all_obs)

        dones = np.asarray(all_term, dtype=bool) | np.asarray(all_trunc, dtype=bool)
        for index, done in enumerate(dones):
            if done:
                all_info[index] = dict(all_info[index])
                all_info[index]["terminal_observation"] = all_obs[index]
                all_info[index]["TimeLimit.truncated"] = bool(all_trunc[index])

        # Return results
        return (
            {k: np.array(v) for k, v in obs.items()},
            np.array(all_rewards, dtype=np.float32),
            dones,
            all_info,
        )

    def reset(self) -> Dict[str, np.ndarray]:
        # Initialize lists for collecting results
        all_obs = []
        all_info = []

        # Reset each environment
        for i in range(self.n_parallel):
            start = i * self.envs[i].num_envs
            seed = self._seeds[start] if self._seeds else None
            obs, info = self.envs[i].reset(seed=seed)
            all_obs.extend(obs)
            all_info.extend(info)

        self.reset_infos = all_info
        self._reset_seeds()
        self._reset_options()

        # Convert list of dictionaries to dictionary of lists
        obs = lod_to_dol(all_obs)
        return {k: np.array(v) for k, v in obs.items()}

    def close(self) -> None:
        # Close each environment
        for env in self.envs:
            env.close()

    def get_images(self) -> List[Any]:
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class: type, indices: Optional[Union[Iterable[int], int]] = None) -> List[bool]:
        # Return a list indicating that no environments are wrapped
        return [False] * len(self._get_indices(indices))

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        return [
            getattr(self.envs[self._env_index(index)], method_name)(*method_args, **method_kwargs)
            for index in self._get_indices(indices)
        ]

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self.envs[self._env_index(index)], attr_name) for index in self._get_indices(indices)]

    def seed(self, seed=None):
        seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        self._seeds = [seed + index for index in range(self.num_envs)]
        return self._seeds

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for index in self._get_indices(indices):
            setattr(self.envs[self._env_index(index)], attr_name, value)

    def _get_indices(self, indices=None) -> Union[range, List[int]]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def _env_index(self, index: int) -> int:
        return index // self.envs[0].num_envs


def stable_baselines_training(args, extras, n_steps: int = 200000, **kwargs) -> None:
    if can_import("ray"):
        print("WARNING, stable baselines and ray[rllib] are not compatible")
    # Initialize the custom environment
    env = StableBaselinesGodotEnv(env_path=args.env_path, show_window=args.viz, speedup=args.speedup, **kwargs)
    env = VecMonitor(env)

    # Initialize the PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.0001,
        verbose=2,
        n_steps=32,
        tensorboard_log=args.experiment_dir or "logs/sb3",
    )

    # Train the model
    model.learn(n_steps, tb_log_name=args.experiment_name)

    print("closing env")
    env.close()
