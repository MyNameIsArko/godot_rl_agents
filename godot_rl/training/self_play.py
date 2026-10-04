from __future__ import annotations

import signal
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger

from godot_rl.core.multi_agent_env import AGENT_IDS, GodotMultiAgentEnv
from godot_rl.wrappers.project_sb3 import _batch_observation, _extract_action_head


class _SpaceOnlyEnv(gym.Env):
    """A space carrier for PPO construction that cannot run training steps."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action: Any):
        raise RuntimeError("the PPO space-only environment cannot produce training steps")

    def close(self) -> None:
        return None


class SelfPlayTrainer:
    """Train two independent PPO policies from one shared environment."""

    agent_ids = AGENT_IDS

    def __init__(
        self,
        env: GodotMultiAgentEnv,
        ppo: Mapping[str, Any],
        *,
        seed: int = 0,
    ) -> None:
        if tuple(env.agent_ids) != AGENT_IDS:
            raise ValueError("self-play requires agent_ids ['player_0', 'player_1']")
        if "n_steps" not in ppo:
            raise ValueError("ppo.n_steps is required")
        if not isinstance(ppo["n_steps"], int) or isinstance(ppo["n_steps"], bool) or ppo["n_steps"] <= 0:
            raise ValueError("ppo.n_steps must be a positive integer")

        self.env = env
        self.seed = seed
        self.n_steps = ppo["n_steps"]
        self.gamma = float(ppo.get("gamma", 0.99))
        self._action_keys: dict[str, str | None] = {}
        self._action_spaces: dict[str, gym.Space] = {}
        self.models: dict[str, PPO] = {}
        self._observations: dict[str, object] | None = None
        self.completed_timesteps = 0
        self.completed_updates = 0
        self._last_completed_timesteps = 0
        self._unsafe_update_interruption = False
        self._interrupted_checkpoint_saved = False
        self._episode_starts = {
            agent_id: np.ones(1, dtype=bool) for agent_id in AGENT_IDS
        }

        for index, agent_id in enumerate(AGENT_IDS):
            action_key, action_space = _extract_action_head(env.action_spaces[agent_id])
            self._action_keys[agent_id] = action_key
            self._action_spaces[agent_id] = action_space
            policy = "MultiInputPolicy" if isinstance(
                env.observation_spaces[agent_id], gym.spaces.Dict
            ) else "MlpPolicy"
            model_kwargs = dict(ppo)
            model_kwargs["seed"] = seed + index
            self.models[agent_id] = PPO(
                policy,
                _SpaceOnlyEnv(env.observation_spaces[agent_id], action_space),
                **model_kwargs,
            )
        self._configure_loggers()

        self._validate_model_spaces()

    def _configure_loggers(self) -> None:
        for model in self.models.values():
            model.set_logger(Logger(folder=None, output_formats=[]))

    def _validate_model_spaces(self) -> None:
        for agent_id in AGENT_IDS:
            model = self.models[agent_id]
            observation_space = model.policy.observation_space
            action_space = model.policy.action_space
            if observation_space != self.env.observation_spaces[agent_id]:
                raise ValueError(f"{agent_id} PPO observation space does not match the live environment")
            if action_space != self._action_spaces[agent_id]:
                raise ValueError(f"{agent_id} PPO action space does not match the live environment")

    def learn(self, target_timesteps: int) -> SelfPlayTrainer:
        if (
            not isinstance(target_timesteps, int)
            or isinstance(target_timesteps, bool)
            or target_timesteps <= 0
            or target_timesteps % self.n_steps
        ):
            raise ValueError("target_timesteps must be a positive multiple of ppo.n_steps")

        current_timesteps = self._current_timesteps()
        if target_timesteps < current_timesteps:
            raise ValueError("target_timesteps must not be less than completed shared timesteps")
        if current_timesteps % self.n_steps:
            raise ValueError("the current shared timestep count must be a multiple of ppo.n_steps")
        if target_timesteps == current_timesteps:
            return self

        if self._observations is None:
            self._observations, _ = self.env.reset()

        self._unsafe_update_interruption = False
        self._interrupted_checkpoint_saved = False
        try:
            while current_timesteps < target_timesteps:
                for model in self.models.values():
                    model.rollout_buffer.reset()
                    model.policy.set_training_mode(False)

                for _ in range(self.n_steps):
                    batched = {
                        agent_id: _batch_observation(
                            self._observations[agent_id], self.env.observation_spaces[agent_id]
                        )
                        for agent_id in AGENT_IDS
                    }
                    actions: dict[str, Any] = {}
                    values: dict[str, torch.Tensor] = {}
                    log_probs: dict[str, torch.Tensor] = {}
                    buffer_actions: dict[str, np.ndarray] = {}

                    with torch.no_grad():
                        for agent_id in AGENT_IDS:
                            model = self.models[agent_id]
                            observation_tensor, _ = model.policy.obs_to_tensor(batched[agent_id])
                            action, value, log_prob = model.policy(observation_tensor)
                            action_array = action.cpu().numpy()
                            buffer_actions[agent_id] = action_array
                            actions[agent_id] = self._environment_action(agent_id, action_array)
                            values[agent_id] = value
                            log_probs[agent_id] = log_prob

                    (
                        next_observations,
                        rewards,
                        terminated,
                        truncated,
                        infos,
                    ) = self.env.step(actions)
                    done = terminated[AGENT_IDS[0]] or truncated[AGENT_IDS[0]]

                    for agent_id in AGENT_IDS:
                        raw_reward = float(rewards[agent_id])
                        reward = raw_reward
                        if truncated[agent_id] and not terminated[agent_id]:
                            terminal_observation = infos[agent_id].get("terminal_observation")
                            if terminal_observation is None:
                                raise ValueError(
                                    f"{agent_id}.info.terminal_observation is required for truncation bootstrap"
                                )
                            terminal_batch = _batch_observation(
                                terminal_observation, self.env.observation_spaces[agent_id]
                            )
                            with torch.no_grad():
                                terminal_tensor, _ = self.models[agent_id].policy.obs_to_tensor(terminal_batch)
                                terminal_value = self.models[agent_id].policy.predict_values(terminal_tensor)
                            reward += self.gamma * float(terminal_value.reshape(-1)[0].item())

                        action_for_buffer = buffer_actions[agent_id]
                        if isinstance(self._action_spaces[agent_id], gym.spaces.Discrete):
                            action_for_buffer = action_for_buffer.reshape(-1, 1)
                        model = self.models[agent_id]
                        model.rollout_buffer.add(
                            batched[agent_id],
                            action_for_buffer,
                            np.asarray([reward], dtype=np.float32),
                            self._episode_starts[agent_id],
                            values[agent_id],
                            log_probs[agent_id],
                        )
                        model.num_timesteps += 1


                    current_timesteps += 1
                    self._observations = next_observations
                    self._episode_starts = {
                        agent_id: np.asarray([done], dtype=bool) for agent_id in AGENT_IDS
                    }
                    if done:
                        self._observations, _ = self.env.reset()

                for agent_id in AGENT_IDS:
                    model = self.models[agent_id]
                    with torch.no_grad():
                        last_batch = _batch_observation(
                            self._observations[agent_id], self.env.observation_spaces[agent_id]
                        )
                        last_tensor, _ = model.policy.obs_to_tensor(last_batch)
                        last_value = model.policy.predict_values(last_tensor)
                    model.rollout_buffer.compute_returns_and_advantage(
                        last_values=last_value,
                        dones=self._episode_starts[agent_id],
                    )
                    model._update_current_progress_remaining(current_timesteps, target_timesteps)

                with self._deferred_sigint() as interrupt_state:
                    self.models[AGENT_IDS[0]].train()
                    self.models[AGENT_IDS[1]].train()
                    interrupt_state["train_complete"] = True
                    self.completed_timesteps = current_timesteps
                    self.completed_updates += 1
                    self._last_completed_timesteps = current_timesteps
                    for model in self.models.values():
                        model.logger.dump(current_timesteps)
                if interrupt_state["requested"]:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            raise

        return self

    @contextmanager
    def _deferred_sigint(self) -> Iterator[dict[str, bool]]:
        state = {"requested": False, "train_complete": False}

        def defer_sigint(signum: int, frame: Any) -> None:
            state["requested"] = True

        if threading.current_thread() is not threading.main_thread():
            try:
                yield state
            except BaseException:
                if not state["train_complete"]:
                    self._unsafe_update_interruption = True
                raise
            return

        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, defer_sigint)
        try:
            try:
                yield state
            except BaseException:
                if not state["train_complete"]:
                    self._unsafe_update_interruption = True
                raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def _current_timesteps(self) -> int:
        values = {self.models[agent_id].num_timesteps for agent_id in AGENT_IDS}
        if len(values) != 1:
            raise ValueError("the two PPO models must have equal shared timestep counters")
        return values.pop()

    def _environment_action(self, agent_id: str, action: np.ndarray) -> object:
        action_space = self._action_spaces[agent_id]
        if isinstance(action_space, gym.spaces.Box):
            if self.models[agent_id].policy.squash_output:
                action = self.models[agent_id].policy.unscale_action(action)
            else:
                action = np.clip(action, action_space.low, action_space.high)
        if isinstance(action_space, gym.spaces.Discrete):
            value: object = int(np.asarray(action).reshape(-1)[0])
        else:
            value = np.asarray(action)[0]
        key = self._action_keys[agent_id]
        return {key: value} if key is not None else value
