import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

from godot_rl import project_cli as cli


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n")
    (project / "rl_training.tscn").write_text("[gd_scene format=3]\n")
    return project


class _ValidationEnv:
    agent_ids = ("player_0", "player_1")
    observation_spaces: ClassVar = {agent_id: gym.spaces.Box(-1, 1, (1,), dtype=np.float32) for agent_id in agent_ids}
    action_spaces: ClassVar = {agent_id: gym.spaces.Discrete(2) for agent_id in agent_ids}

    def __init__(self, completes: bool):
        self.completes = completes
        self.closed = False
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed=None):
        self.reset_count += 1
        self.step_count = 0
        return (
            {agent_id: np.zeros(1, dtype=np.float32) for agent_id in self.agent_ids},
            {agent_id: {} for agent_id in self.agent_ids},
        )

    def step(self, actions):
        self.step_count += 1
        done = self.completes and self.step_count == 1
        infos = {
            "player_0": {"outcome": "win", "terminal_observation": np.zeros(1, dtype=np.float32)} if done else {},
            "player_1": {"outcome": "loss", "terminal_observation": np.zeros(1, dtype=np.float32)} if done else {},
        }
        return (
            {agent_id: np.zeros(1, dtype=np.float32) for agent_id in self.agent_ids},
            {agent_id: 1.0 for agent_id in self.agent_ids},
            {agent_id: done for agent_id in self.agent_ids},
            {agent_id: False for agent_id in self.agent_ids},
            infos,
        )

    def close(self):
        self.closed = True


class _BoxEvaluationEnv(_ValidationEnv):
    observation_spaces: ClassVar = {
        agent_id: gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        for agent_id in _ValidationEnv.agent_ids
    }
    action_spaces: ClassVar = {
        agent_id: gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        for agent_id in _ValidationEnv.agent_ids
    }

    def __init__(self):
        super().__init__(completes=True)
        self.step_batches = []

    def reset(self, *, seed=None):
        self.reset_count += 1
        self.step_count = 0
        return (
            {agent_id: np.zeros(2, dtype=np.float32) for agent_id in self.agent_ids},
            {agent_id: {} for agent_id in self.agent_ids},
        )

    def step(self, actions):
        self.step_batches.append(actions)
        self.step_count += 1
        done = self.step_count == 2
        infos = {
            "player_0": {"outcome": "win", "terminal_observation": np.zeros(2, dtype=np.float32)} if done else {},
            "player_1": {"outcome": "loss", "terminal_observation": np.zeros(2, dtype=np.float32)} if done else {},
        }
        reward = float(self.step_count)
        return (
            {agent_id: np.zeros(2, dtype=np.float32) for agent_id in self.agent_ids},
            {agent_id: reward for agent_id in self.agent_ids},
            {agent_id: done for agent_id in self.agent_ids},
            {agent_id: False for agent_id in self.agent_ids},
            infos,
        )


def _v2_project(tmp_path):
    project = make_project(tmp_path)
    (project / "rl").mkdir()
    (project / "rl/config.toml").write_text(cli.SELF_PLAY_CONFIG_TEXT)
    return project


def test_self_play_validation_requires_a_completed_episode(tmp_path, monkeypatch):
    project = _v2_project(tmp_path)
    env = _ValidationEnv(completes=False)
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ok": True, "godot": {"path": "Godot"}})
    monkeypatch.setattr(cli, "GodotMultiAgentEnv", lambda **kwargs: env)
    with pytest.raises(ValueError, match="increase --steps or reduce"):
        cli.validate_project(project, 2)
    assert env.closed is True


def test_train_rejects_schema_v2_actionably(tmp_path):
    project = _v2_project(tmp_path)
    with pytest.raises(ValueError, match="use gdrl self-play"):
        cli.train_project(project, 128, "old")


def test_self_play_rejects_schema_v1_before_doctor(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    (project / "rl").mkdir()
    (project / "rl/config.toml").write_text(cli.CONFIG_TEXT)
    called = []
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="gdrl self-play requires schema version 2"):
        cli.self_play_project(project, 128, "duel")
    assert called == []


def test_evaluate_rejects_schema_v1_before_doctor(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    (project / "rl").mkdir()
    (project / "rl/config.toml").write_text(cli.CONFIG_TEXT)
    called = []
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="gdrl evaluate requires schema version 2"):
        cli.evaluate_project(project, str(tmp_path / "checkpoint"), 2)
    assert called == []


def test_self_play_rejects_unsafe_run_names(tmp_path):
    for name in (".hidden", "bad name", "nested/run", "nested\\run"):
        with pytest.raises(ValueError, match="portable safe"):
            cli.self_play_project(tmp_path, 128, name)


def test_self_play_rejects_resume_checkpoint_outside_project_layout(tmp_path, monkeypatch):
    project = _v2_project(tmp_path)
    outside = tmp_path / "outside-checkpoint"
    outside.mkdir()
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ok": True, "godot": {"path": "Godot"}})
    with pytest.raises(ValueError, match="under project/rl/models/<name>/checkpoints"):
        cli.self_play_project(project, 128, "duel", resume=str(outside))


@pytest.mark.parametrize("speedup", [-1, 0, float("nan"), float("inf"), float("-inf")])
def test_self_play_rejects_invalid_speedup_before_doctor(tmp_path, monkeypatch, speedup):
    called = []
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="positive finite"):
        cli.self_play_project(tmp_path, 128, "duel", speedup=speedup)
    assert called == []


@pytest.mark.parametrize("speedup", [-1, 0, float("nan"), float("inf"), float("-inf")])
def test_evaluate_rejects_invalid_speedup_before_doctor(tmp_path, monkeypatch, speedup):
    called = []
    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="positive finite"):
        cli.evaluate_project(tmp_path, "checkpoint", 2, speedup=speedup)
    assert called == []


def test_self_play_command_trains_and_persists_final_pair(tmp_path, monkeypatch):
    project = _v2_project(tmp_path)
    env = _ValidationEnv(completes=True)
    seen = {}

    class FakeTrainer:
        def __init__(self, environment, ppo, **kwargs):
            seen["ppo"] = ppo
            seen["kwargs"] = kwargs
            self.completed_timesteps = 0
            self._interrupted_checkpoint_saved = False
            self._unsafe_update_interruption = False

        @staticmethod
        def configuration_hash(path):
            return "0" * 64

        def learn(self, target):
            self.completed_timesteps = target

        @staticmethod
        def _checkpoint_complete(path):
            return False

        def save_checkpoint(self):
            checkpoint = Path(seen["kwargs"]["model_dir"]) / "checkpoints" / f"{self.completed_timesteps:012d}"
            checkpoint.mkdir(parents=True)
            for filename in ("player_0.zip", "player_1.zip", "state.json"):
                (checkpoint / filename).write_text("{}")
            return checkpoint

    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ok": True, "godot": {"path": "Godot"}})
    monkeypatch.setattr(cli, "GodotMultiAgentEnv", lambda **kwargs: env)
    monkeypatch.setattr("godot_rl.training.self_play.SelfPlayTrainer", FakeTrainer)
    assert cli.self_play_project(project, 128, "duel") == 0
    assert seen["ppo"]["n_steps"] == 128
    assert seen["kwargs"]["run_name"] == "duel"
    assert env.closed is True
    assert (project / "rl/models/duel/checkpoints/000000000128").is_dir()


def test_evaluate_is_deterministic_seat_swapped_and_read_only(tmp_path, monkeypatch, capsys):
    project = _v2_project(tmp_path)
    checkpoint = project / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text('{"run_name":"duel"}')
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    env = _ValidationEnv(completes=True)
    env.step_batches = []

    def step(actions):
        env.step_batches.append(actions)
        return _ValidationEnv.step(env, actions)

    env.step = step

    class Model:
        def __init__(self, policy_id):
            self.policy_id = policy_id

        def predict(self, observation, deterministic=False):
            assert deterministic is True
            return self.policy_id, None

    class FakeTrainer:
        models: ClassVar = {"player_0": Model(0), "player_1": Model(1)}

        @staticmethod
        def configuration_hash(path):
            return "0" * 64

        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            return cls()

        @staticmethod
        def _environment_action(agent_id, action):
            return int(action)

    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ok": True, "godot": {"path": "Godot"}})
    monkeypatch.setattr(cli, "GodotMultiAgentEnv", lambda **kwargs: env)
    monkeypatch.setattr("godot_rl.training.self_play.SelfPlayTrainer", FakeTrainer)
    assert cli.evaluate_project(project, str(checkpoint), 4, max_steps=2) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["seats"] == {"normal_episodes": 2, "swapped_episodes": 2}
    assert result["policy_0"]["wins"] == result["policy_0"]["losses"] == 2
    assert env.step_batches[0] == {"player_0": 0, "player_1": 1}
    assert env.step_batches[2] == {"player_0": 1, "player_1": 0}
    assert before == {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    assert env.closed is True


def test_evaluate_batches_box_actions_swaps_complete_vectors_and_averages_returns(
    tmp_path, monkeypatch, capsys
):
    project = _v2_project(tmp_path)
    checkpoint = project / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text('{"run_name":"duel"}')
    env = _BoxEvaluationEnv()

    class Model:
        def __init__(self, action):
            self.action = np.asarray([action], dtype=np.float32)

        def predict(self, observation, deterministic=False):
            assert deterministic is True
            assert np.asarray(observation).shape == (1, 2)
            return self.action, None

    class FakeTrainer:
        models: ClassVar = {
            "player_0": Model([0.1, 0.2]),
            "player_1": Model([0.7, 0.8]),
        }

        @staticmethod
        def configuration_hash(path):
            return "0" * 64

        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            return cls()

        @staticmethod
        def _environment_action(agent_id, action):
            action = np.asarray(action)
            assert action.shape == (1, 2)
            return action[0]

    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ok": True, "godot": {"path": "Godot"}})
    monkeypatch.setattr(cli, "GodotMultiAgentEnv", lambda **kwargs: env)
    monkeypatch.setattr("godot_rl.training.self_play.SelfPlayTrainer", FakeTrainer)
    assert cli.evaluate_project(project, str(checkpoint), 4, max_steps=2) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["policy_0"]["mean_reward"] == 3.0
    assert result["policy_1"]["mean_reward"] == 3.0
    np.testing.assert_allclose(env.step_batches[0]["player_0"], [0.1, 0.2])
    np.testing.assert_allclose(env.step_batches[4]["player_0"], [0.7, 0.8])


def test_evaluate_requires_an_even_episode_count(tmp_path):
    with pytest.raises(ValueError, match="positive even"):
        cli.evaluate_project(tmp_path, "checkpoint", 3)
