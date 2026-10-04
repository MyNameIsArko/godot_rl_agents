import inspect
import json
import signal
import threading
from typing import ClassVar, get_args, get_type_hints

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

import godot_rl.training.self_play as self_play_module
from godot_rl.training.self_play import SelfPlayTrainer, _SpaceOnlyEnv

OBSERVATION_SPACE = gym.spaces.Dict({"obs": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)})
ACTION_SPACE = gym.spaces.Dict({"action": gym.spaces.Discrete(2)})


class FakeEnv:
    agent_ids = ("player_0", "player_1")
    observation_spaces: ClassVar = {agent_id: OBSERVATION_SPACE for agent_id in agent_ids}
    action_spaces: ClassVar = {agent_id: ACTION_SPACE for agent_id in agent_ids}

    def __init__(self, *, truncated=False, terminated=False):
        self.truncated = truncated
        self.terminated = terminated
        self.reset_count = 0
        self.step_count = 0
        self.action_batches = []
        self.events = []

    def reset(self):
        self.reset_count += 1
        self.events.append("reset")
        return (
            {agent_id: {"obs": np.zeros(2, dtype=np.float32)} for agent_id in self.agent_ids},
            {agent_id: {} for agent_id in self.agent_ids},
        )

    def step(self, actions):
        assert set(actions) == set(self.agent_ids)
        self.action_batches.append(actions)
        self.step_count += 1
        self.events.append("step")
        done = self.truncated or self.terminated
        infos = {
            agent_id: (
                {
                    "terminal_observation": {"obs": np.zeros(2, dtype=np.float32)},
                    "outcome": "win" if agent_id == "player_0" else "loss",
                }
                if done
                else {}
            )
            for agent_id in self.agent_ids
        }
        return (
            {agent_id: {"obs": np.zeros(2, dtype=np.float32)} for agent_id in self.agent_ids},
            {agent_id: 1.0 for agent_id in self.agent_ids},
            {agent_id: self.terminated for agent_id in self.agent_ids},
            {agent_id: self.truncated for agent_id in self.agent_ids},
            infos,
        )


class BoxObservationEnv(FakeEnv):
    observation_spaces: ClassVar = {
        agent_id: gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32) for agent_id in FakeEnv.agent_ids
    }


def make_trainer(*, n_steps=2, truncated=False, terminated=False, **kwargs):
    return SelfPlayTrainer(
        FakeEnv(truncated=truncated, terminated=terminated),
        {
            "n_steps": n_steps,
            "batch_size": n_steps,
            "n_epochs": 1,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        seed=17,
        **kwargs,
    )


def test_space_only_env_cannot_step():
    env = _SpaceOnlyEnv(OBSERVATION_SPACE, gym.spaces.Discrete(2))
    env.reset()
    with pytest.raises(RuntimeError, match="cannot produce training steps"):
        env.step(0)


def test_models_are_independent_and_seeded_differently():
    trainer = make_trainer()
    left = trainer.models["player_0"]
    right = trainer.models["player_1"]
    assert left.seed == 17
    assert right.seed == 18
    assert left.n_envs == right.n_envs == 1
    assert isinstance(left.policy, MultiInputActorCriticPolicy)
    assert isinstance(right.policy, MultiInputActorCriticPolicy)
    assert next(left.policy.parameters()).data_ptr() != next(right.policy.parameters()).data_ptr()


def test_direct_box_observations_select_mlp_policy():
    trainer = SelfPlayTrainer(
        BoxObservationEnv(),
        {
            "n_steps": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    )
    assert all(model.n_envs == 1 for model in trainer.models.values())
    assert all(isinstance(model.policy, ActorCriticPolicy) for model in trainer.models.values())
    assert all(not isinstance(model.policy, MultiInputActorCriticPolicy) for model in trainer.models.values())


def test_shared_rollout_updates_both_models_once():
    trainer = make_trainer()
    train_calls = []

    for agent_id, model in trainer.models.items():
        original = model.train

        def train(agent_id=agent_id, original=original):
            assert all(
                buffer.full and buffer.pos == trainer.n_steps
                for buffer in (candidate.rollout_buffer for candidate in trainer.models.values())
            )
            train_calls.append(agent_id)
            trainer.env.events.append("train")
            return original()

        model.train = train

    trainer.learn(2)
    assert trainer.env.step_count == 2
    assert [set(batch) for batch in trainer.env.action_batches] == [
        {"player_0", "player_1"},
        {"player_0", "player_1"},
    ]
    assert trainer.env.events[-4:] == ["step", "step", "train", "train"]
    assert all(
        model.rollout_buffer.full and model.rollout_buffer.pos == trainer.n_steps for model in trainer.models.values()
    )
    assert train_calls == ["player_0", "player_1"]
    assert [model.num_timesteps for model in trainer.models.values()] == [2, 2]


def test_both_policies_produce_outputs_before_each_shared_step():
    trainer = make_trainer()
    outputs = {agent_id: [] for agent_id in trainer.agent_ids}

    for agent_id, model in trainer.models.items():
        original = model.policy.forward

        def forward(*args, agent_id=agent_id, original=original, **kwargs):
            result = original(*args, **kwargs)
            action, value, log_prob = result
            outputs[agent_id].append((action.detach(), value.detach(), log_prob.detach()))
            trainer.env.events.append(f"{agent_id}:policy")
            return result

        model.policy.forward = forward

    trainer.learn(4)

    assert all(len(agent_outputs) == 4 for agent_outputs in outputs.values())
    assert all(
        action.shape[0] == value.shape[0] == log_prob.shape[0] == 1
        for agent_outputs in outputs.values()
        for action, value, log_prob in agent_outputs
    )
    events = [event for event in trainer.env.events if event != "reset"]
    assert (
        events
        == [
            "player_0:policy",
            "player_1:policy",
            "step",
        ]
        * 4
    )


def test_buffer_values_are_collected_before_policy_updates():
    trainer = make_trainer()
    captured_values = {agent_id: [] for agent_id in trainer.agent_ids}
    train_calls = []

    for agent_id, model in trainer.models.items():
        original_forward = model.policy.forward

        def forward(*args, agent_id=agent_id, original=original_forward, **kwargs):
            result = original(*args, **kwargs)
            captured_values[agent_id].append(result[1].detach().cpu().numpy().copy())
            return result

        model.policy.forward = forward
        original_train = model.train

        def train(agent_id=agent_id, original=original_train):
            assert all(len(values) == trainer.n_steps for values in captured_values.values())
            for captured_agent_id in trainer.agent_ids:
                np.testing.assert_allclose(
                    trainer.models[captured_agent_id].rollout_buffer.values[:, 0],
                    np.asarray([value[0, 0] for value in captured_values[captured_agent_id]]),
                )
            train_calls.append(agent_id)
            return original()

        model.train = train

    trainer.learn(2)

    assert train_calls == ["player_0", "player_1"]
    for agent_id in trainer.agent_ids:
        np.testing.assert_allclose(
            trainer.models[agent_id].rollout_buffer.values[:, 0],
            np.asarray([value[0, 0] for value in captured_values[agent_id]]),
        )


def test_episode_boundary_resets_shared_environment_once():
    trainer = make_trainer(truncated=True)
    trainer.learn(2)
    assert trainer.env.reset_count == 3
    assert all(starts.tolist() == [True] for starts in trainer._episode_starts.values())


def test_truncation_bootstraps_each_agent_reward(monkeypatch):
    trainer = make_trainer(truncated=True)
    for model in trainer.models.values():
        monkeypatch.setattr(model.policy, "predict_values", lambda observation: torch.tensor([[2.0]]))
    trainer.learn(2)
    for model in trainer.models.values():
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [2.98, 2.98])


def test_termination_does_not_bootstrap_terminal_value(monkeypatch):
    trainer = make_trainer(terminated=True)
    for model in trainer.models.values():
        monkeypatch.setattr(model.policy, "predict_values", lambda observation: torch.tensor([[2.0]]))
    trainer.learn(2)
    for model in trainer.models.values():
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [1.0, 1.0])


def test_partial_rollout_request_fails():
    trainer = make_trainer(n_steps=2)
    with pytest.raises(ValueError, match="positive multiple"):
        trainer.learn(3)


def test_compatibility_signatures_match_supported_sb3_versions():
    assert stable_baselines3.__version__ in {"2.4.0", "2.9.0"}

    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    var_positional = inspect.Parameter.VAR_POSITIONAL
    var_keyword = inspect.Parameter.VAR_KEYWORD

    def assert_signature(method, names, *, kinds=None, defaults=None):
        signature = inspect.signature(method)
        assert tuple(signature.parameters) == tuple(names)
        expected_kinds = kinds or (positional,) * len(names)
        assert tuple(parameter.kind for parameter in signature.parameters.values()) == tuple(expected_kinds)
        for name, expected_default in (defaults or {}).items():
            assert signature.parameters[name].default == expected_default
        return signature

    assert_signature(
        PPO.__init__,
        (
            "self",
            "policy",
            "env",
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "clip_range_vf",
            "normalize_advantage",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "use_sde",
            "sde_sample_freq",
            "rollout_buffer_class",
            "rollout_buffer_kwargs",
            "target_kl",
            "stats_window_size",
            "tensorboard_log",
            "policy_kwargs",
            "verbose",
            "seed",
            "device",
            "_init_setup_model",
        ),
        defaults={
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": None,
            "_init_setup_model": True,
        },
    )
    assert_signature(BasePolicy.__call__, ("self", "args", "kwargs"), kinds=(positional, var_positional, var_keyword))
    assert_signature(BasePolicy.obs_to_tensor, ("self", "observation"))
    assert_signature(BasePolicy.set_training_mode, ("self", "mode"))
    assert_signature(BasePolicy.unscale_action, ("self", "scaled_action"))
    assert_signature(ActorCriticPolicy.forward, ("self", "obs", "deterministic"), defaults={"deterministic": False})
    assert_signature(ActorCriticPolicy.predict_values, ("self", "obs"))
    assert_signature(RolloutBuffer.reset, ("self",))
    assert_signature(RolloutBuffer.add, ("self", "obs", "action", "reward", "episode_start", "value", "log_prob"))
    assert_signature(RolloutBuffer.compute_returns_and_advantage, ("self", "last_values", "dones"))
    assert_signature(PPO._update_current_progress_remaining, ("self", "num_timesteps", "total_timesteps"))
    assert_signature(PPO.train, ("self",))
    assert_signature(PPO.set_logger, ("self", "logger"))
    assert_signature(Logger.dump, ("self", "step"), defaults={"step": 0})
    assert_signature(PPO.save, ("self", "path", "exclude", "include"), defaults={"exclude": None, "include": None})
    assert_signature(
        PPO.load,
        ("path", "env", "device", "custom_objects", "print_system_info", "force_reset", "kwargs"),
        kinds=(positional, positional, positional, positional, positional, positional, var_keyword),
        defaults={
            "env": None,
            "device": "auto",
            "custom_objects": None,
            "print_system_info": False,
            "force_reset": True,
        },
    )

    assert len(get_args(get_type_hints(BasePolicy.obs_to_tensor)["return"])) == 2
    assert len(get_args(get_type_hints(ActorCriticPolicy.forward)["return"])) == 3
