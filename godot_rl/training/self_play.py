from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger, configure

from godot_rl.core.multi_agent_env import AGENT_IDS, GodotMultiAgentEnv
from godot_rl.wrappers.project_sb3 import _batch_observation, _extract_action_head

_CHECKPOINT_FILES = frozenset({"player_0.zip", "player_1.zip", "state.json"})
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_NAME_RE = re.compile(r"[0-9]{12}\Z")


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
        model_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        run_name: str | None = None,
        configuration_sha256: str | None = None,
        checkpoint_interval: int | None = None,
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
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir is not None else None
        self.run_dir = Path(run_dir).expanduser().resolve() if run_dir is not None else None
        self.run_name = run_name
        self.configuration_sha256 = configuration_sha256
        self.checkpoint_interval = checkpoint_interval
        if self.run_name is not None:
            self._validate_run_name(self.run_name)
        if configuration_sha256 is not None:
            self._validate_hash(configuration_sha256)
        if checkpoint_interval is not None and (
            not isinstance(checkpoint_interval, int)
            or isinstance(checkpoint_interval, bool)
            or checkpoint_interval <= 0
            or checkpoint_interval % self.n_steps
        ):
            raise ValueError("checkpoint_interval must be a positive multiple of ppo.n_steps")
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

    @staticmethod
    def _validate_run_name(run_name: str) -> None:
        if (
            not run_name
            or "/" in run_name
            or "\\" in run_name
            or run_name in {".", ".."}
        ):
            raise ValueError("run_name must be a single safe path component")

    @staticmethod
    def _validate_hash(value: str) -> None:
        if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
            raise ValueError("configuration_sha256 must be 64 lowercase hexadecimal characters")

    @staticmethod
    def configuration_hash(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _package_version() -> str:
        return importlib.metadata.version("godot_rl")

    def _configure_loggers(self) -> None:
        if self.run_dir is None:
            logger = Logger(folder=None, output_formats=[])
            for model in self.models.values():
                model.set_logger(logger)
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        output_formats = ["csv"]
        try:
            __import__("tensorboard")
        except ImportError:
            pass
        else:
            output_formats.append("tensorboard")
        for agent_id in AGENT_IDS:
            self.models[agent_id].set_logger(
                configure(str(self.run_dir / agent_id), output_formats)
            )

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

                rollout_stats = self._new_rollout_stats()

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
                        rollout_stats[agent_id]["reward"] += raw_reward
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

                        if done:
                            rollout_stats[agent_id]["episodes"] += 1
                            outcome = infos[agent_id].get("outcome")
                            if outcome in {"win", "loss", "draw"}:
                                rollout_stats[agent_id][outcome] += 1

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
                    self._append_rollout_metrics(current_timesteps, rollout_stats)
                    if interrupt_state["requested"]:
                        if (
                            self.model_dir is not None
                            and self.run_name is not None
                            and self.configuration_sha256 is not None
                        ):
                            self.save_interrupted_checkpoint()
                        self._interrupted_checkpoint_saved = True
                    elif self.checkpoint_interval and current_timesteps % self.checkpoint_interval == 0:
                        self.save_checkpoint()
                if interrupt_state["requested"]:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            if (
                not self._unsafe_update_interruption
                and not self._interrupted_checkpoint_saved
                and self.model_dir is not None
                and self.run_name is not None
                and self.configuration_sha256 is not None
            ):
                self.save_interrupted_checkpoint()
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

    @staticmethod
    def _new_rollout_stats() -> dict[str, dict[str, float | int]]:
        return {
            agent_id: {"reward": 0.0, "win": 0, "loss": 0, "draw": 0, "episodes": 0}
            for agent_id in AGENT_IDS
        }

    def _append_rollout_metrics(
        self, completed_timesteps: int, stats: Mapping[str, Mapping[str, float | int]]
    ) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        episodes = int(stats[AGENT_IDS[0]]["episodes"])
        if any(int(stats[agent_id]["episodes"]) != episodes for agent_id in AGENT_IDS[1:]):
            raise ValueError("both policies must report the same completed episode count")
        record: dict[str, Any] = {
            "type": "rollout",
            "completed_timesteps": completed_timesteps,
            "completed_updates": self.completed_updates,
            "episodes": episodes,
        }
        for agent_id in AGENT_IDS:
            agent_stats = stats[agent_id]
            record[agent_id] = {
                "mean_reward": float(agent_stats["reward"]) / self.n_steps,
                "wins": int(agent_stats["win"]),
                "losses": int(agent_stats["loss"]),
                "draws": int(agent_stats["draw"]),
            }
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
            json.dump(record, metrics, separators=(",", ":"))
            metrics.write("\n")
            metrics.flush()
            os.fsync(metrics.fileno())

    def _checkpoint_context(
        self,
        model_dir: str | Path | None,
        run_name: str | None,
        configuration_sha256: str | None,
    ) -> tuple[Path, str, str]:
        target_dir = Path(model_dir).expanduser().resolve() if model_dir is not None else self.model_dir
        if target_dir is None:
            raise ValueError("model_dir is required for checkpoint persistence")
        target_name = run_name if run_name is not None else self.run_name
        if target_name is None:
            raise ValueError("run_name is required for checkpoint persistence")
        self._validate_run_name(target_name)
        config_hash = configuration_sha256 if configuration_sha256 is not None else self.configuration_sha256
        if config_hash is None:
            raise ValueError("configuration_sha256 is required for checkpoint persistence")
        self._validate_hash(config_hash)
        return target_dir, target_name, config_hash

    @staticmethod
    def _checkpoint_complete(checkpoint: Path) -> bool:
        return (
            checkpoint.is_dir()
            and {path.name for path in checkpoint.iterdir()} == _CHECKPOINT_FILES
            and all((checkpoint / name).is_file() for name in _CHECKPOINT_FILES)
        )

    def save_checkpoint(
        self,
        model_dir: str | Path | None = None,
        *,
        run_name: str | None = None,
        configuration_sha256: str | None = None,
        seed: int | None = None,
        completed_timesteps: int | None = None,
        completed_updates: int | None = None,
    ) -> Path:
        target_dir, target_name, config_hash = self._checkpoint_context(
            model_dir, run_name, configuration_sha256
        )
        timesteps = self._current_timesteps() if completed_timesteps is None else completed_timesteps
        updates = self.completed_updates if completed_updates is None else completed_updates
        if (
            not isinstance(timesteps, int)
            or isinstance(timesteps, bool)
            or timesteps < 0
            or timesteps % self.n_steps
        ):
            raise ValueError("completed_timesteps must be a non-negative multiple of ppo.n_steps")
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
            raise ValueError("completed_updates must be a non-negative integer")
        if updates != timesteps // self.n_steps:
            raise ValueError("completed_updates must match completed_timesteps")
        if timesteps != self._current_timesteps():
            raise ValueError("both PPO models must match the checkpoint timestep")

        checkpoints = target_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoints / f"{timesteps:012d}"
        if checkpoint.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint.name}-", dir=checkpoints))
        try:
            for agent_id in AGENT_IDS:
                self.models[agent_id].save(str(temporary / agent_id))
                model_file = temporary / f"{agent_id}.zip"
                if not model_file.is_file():
                    raise OSError(f"PPO did not write {model_file.name}")
            state = {
                "schema_version": 1,
                "package_version": self._package_version(),
                "run_name": target_name,
                "agent_ids": list(AGENT_IDS),
                "completed_timesteps": timesteps,
                "completed_updates": updates,
                "seed": self.seed if seed is None else seed,
                "configuration_sha256": config_hash,
                "models": {agent_id: f"{agent_id}.zip" for agent_id in AGENT_IDS},
            }
            with (temporary / "state.json").open("x", encoding="utf-8") as state_file:
                json.dump(state, state_file, indent=2, sort_keys=True)
                state_file.write("\n")
                state_file.flush()
                os.fsync(state_file.fileno())
            os.rename(temporary, checkpoint)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._write_latest(target_dir, checkpoint)
        self.model_dir = target_dir
        self.run_name = target_name
        self.configuration_sha256 = config_hash
        return checkpoint

    def save_interrupted_checkpoint(self) -> Path:
        if self.model_dir is None or self.run_name is None:
            raise ValueError("model_dir and run_name are required for interrupted checkpoints")
        current = self._current_timesteps()
        completed = self._last_completed_timesteps
        if current == completed:
            try:
                return self.save_checkpoint()
            except FileExistsError:
                checkpoint = self.model_dir / "checkpoints" / f"{completed:012d}"
                if self._checkpoint_complete(checkpoint):
                    self._write_latest(self.model_dir, checkpoint)
                    return checkpoint
                raise
        original = {agent_id: self.models[agent_id].num_timesteps for agent_id in AGENT_IDS}
        try:
            for agent_id in AGENT_IDS:
                self.models[agent_id].num_timesteps = completed
            try:
                return self.save_checkpoint(completed_timesteps=completed, completed_updates=completed // self.n_steps)
            except FileExistsError:
                checkpoint = self.model_dir / "checkpoints" / f"{completed:012d}"
                if self._checkpoint_complete(checkpoint):
                    self._write_latest(self.model_dir, checkpoint)
                    return checkpoint
                raise
        finally:
            for agent_id, value in original.items():
                self.models[agent_id].num_timesteps = value

    @staticmethod
    def _write_latest(model_dir: Path, checkpoint: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        pointer = checkpoint.relative_to(model_dir).as_posix()
        latest = model_dir / "latest.json"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=model_dir, prefix=".latest-", delete=False
            ) as temporary:
                temporary_name = temporary.name
                json.dump({"schema_version": 1, "checkpoint": pointer}, temporary)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, latest)
        except BaseException:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def latest_checkpoint(model_dir: str | Path) -> Path:
        model_root = Path(model_dir).expanduser().resolve()
        latest_path = model_root / "latest.json"
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid latest checkpoint pointer: {latest_path}") from exc
        if (
            not isinstance(latest, dict)
            or set(latest) != {"schema_version", "checkpoint"}
            or latest["schema_version"] != 1
            or not isinstance(latest["checkpoint"], str)
        ):
            raise ValueError("latest.json has an invalid shape")
        checkpoint = (model_root / latest["checkpoint"]).resolve()
        if checkpoint.parent != model_root / "checkpoints" or not _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name):
            raise ValueError("latest.json points outside the checkpoint directory")
        if not SelfPlayTrainer._checkpoint_complete(checkpoint):
            raise ValueError("latest.json points to an incomplete checkpoint")
        return checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        env: GodotMultiAgentEnv,
        ppo: Mapping[str, Any],
        checkpoint: str | Path,
        *,
        run_name: str,
        configuration_sha256: str,
        seed: int = 0,
        model_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        checkpoint_interval: int | None = None,
    ) -> SelfPlayTrainer:
        trainer = cls(
            env,
            ppo,
            seed=seed,
            model_dir=model_dir or Path(checkpoint).expanduser().resolve().parent.parent,
            run_dir=run_dir,
            run_name=run_name,
            configuration_sha256=configuration_sha256,
            checkpoint_interval=checkpoint_interval,
        )
        trainer.load_checkpoint(
            checkpoint,
            run_name=run_name,
            configuration_sha256=configuration_sha256,
        )
        return trainer

    def load_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        run_name: str | None = None,
        configuration_sha256: str | None = None,
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not _CHECKPOINT_NAME_RE.fullmatch(checkpoint_path.name):
            raise ValueError("checkpoint directory must use twelve decimal digits")
        if not self._checkpoint_complete(checkpoint_path):
            raise ValueError(f"checkpoint is incomplete: {checkpoint_path}")
        state_path = checkpoint_path / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint state: {state_path}") from exc
        self._validate_state(state)
        completed_timesteps = state["completed_timesteps"]
        if completed_timesteps % self.n_steps or state["completed_updates"] != completed_timesteps // self.n_steps:
            raise ValueError("checkpoint counters are inconsistent with ppo.n_steps")
        if int(checkpoint_path.name) != completed_timesteps:
            raise ValueError("checkpoint directory name does not match completed_timesteps")
        expected_name = run_name if run_name is not None else self.run_name
        if expected_name is None:
            raise ValueError("run_name is required to resume a checkpoint")
        config_hash = (
            configuration_sha256
            if configuration_sha256 is not None
            else self.configuration_sha256
        )
        if config_hash is None:
            raise ValueError("configuration_sha256 is required to resume a checkpoint")
        self._validate_hash(config_hash)
        if state["run_name"] != expected_name:
            raise ValueError("checkpoint run_name does not match the requested run")
        if state["configuration_sha256"] != config_hash:
            raise ValueError("checkpoint configuration_sha256 does not match the current configuration")
        if state["package_version"] != self._package_version():
            raise ValueError("checkpoint package_version does not match the installed package")

        loaded: dict[str, PPO] = {}
        for agent_id in AGENT_IDS:
            action_space = self._action_spaces[agent_id]
            loaded[agent_id] = PPO.load(
                str(checkpoint_path / state["models"][agent_id]),
                env=_SpaceOnlyEnv(self.env.observation_spaces[agent_id], action_space),
                force_reset=False,
            )
        self.models = loaded
        self._configure_loggers()
        self._validate_model_spaces()
        if any(model.num_timesteps != state["completed_timesteps"] for model in self.models.values()):
            raise ValueError("checkpoint model timestep counters do not match state.json")
        self.completed_timesteps = state["completed_timesteps"]
        self.completed_updates = state["completed_updates"]
        self.seed = state["seed"]
        self._last_completed_timesteps = self.completed_timesteps
        self._observations = None
        self._episode_starts = {
            agent_id: np.ones(1, dtype=bool) for agent_id in AGENT_IDS
        }
        self.model_dir = checkpoint_path.parent.parent
        self.run_name = expected_name
        self.configuration_sha256 = config_hash
        return state

    @staticmethod
    def _validate_state(state: Any) -> None:
        required = {
            "schema_version", "package_version", "run_name", "agent_ids",
            "completed_timesteps", "completed_updates", "seed",
            "configuration_sha256", "models",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("checkpoint state.json has an invalid shape")
        if state["schema_version"] != 1:
            raise ValueError("checkpoint state schema_version must be 1")
        if not isinstance(state["package_version"], str) or not state["package_version"]:
            raise ValueError("checkpoint package_version is invalid")
        if state["agent_ids"] != list(AGENT_IDS):
            raise ValueError("checkpoint agent_ids must be ['player_0', 'player_1']")
        for key in ("completed_timesteps", "completed_updates", "seed"):
            if not isinstance(state[key], int) or isinstance(state[key], bool):
                raise TypeError(f"checkpoint {key} must be an integer")
        if state["completed_timesteps"] < 0 or state["completed_updates"] < 0:
            raise ValueError("checkpoint counters must be non-negative")
        SelfPlayTrainer._validate_hash(state["configuration_sha256"])
        if not isinstance(state["run_name"], str):
            raise TypeError("checkpoint run_name is invalid")
        SelfPlayTrainer._validate_run_name(state["run_name"])
        if state["models"] != {agent_id: f"{agent_id}.zip" for agent_id in AGENT_IDS}:
            raise ValueError("checkpoint models must contain both paired model files")

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
