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


def test_checkpoint_contains_two_models_and_one_state_file(tmp_path):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)

    checkpoint = trainer.save_checkpoint()
    assert checkpoint.name == "000000000002"
    assert {path.name for path in checkpoint.iterdir()} == {"player_0.zip", "player_1.zip", "state.json"}
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["completed_timesteps"] == 2
    assert state["completed_updates"] == 1
    assert state["models"] == {"player_0": "player_0.zip", "player_1": "player_1.zip"}


def test_checkpoint_is_published_only_after_all_files_complete(tmp_path, monkeypatch):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    final = tmp_path / "models" / "duel" / "checkpoints" / "000000000002"
    original_save = trainer.models["player_0"].save

    def save(path):
        original_save(path)
        assert not final.exists()

    monkeypatch.setattr(trainer.models["player_0"], "save", save)
    checkpoint = trainer.save_checkpoint()
    assert checkpoint == final
    assert final.is_dir()


def test_latest_points_to_complete_checkpoint(tmp_path):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()

    latest = json.loads((tmp_path / "models" / "duel" / "latest.json").read_text())
    assert latest == {"schema_version": 1, "checkpoint": "checkpoints/000000000002"}
    assert trainer._checkpoint_complete(tmp_path / "models" / "duel" / latest["checkpoint"])
    assert trainer.latest_checkpoint(tmp_path / "models" / "duel") == checkpoint
    assert checkpoint.is_dir()


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}

    with pytest.raises(FileExistsError, match="already exists"):
        trainer.save_checkpoint()
    assert before == {path.name: path.read_bytes() for path in checkpoint.iterdir()}


def test_resume_restores_both_models_and_counters(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()

    resumed = SelfPlayTrainer.from_checkpoint(
        FakeEnv(),
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
        checkpoint,
        run_name="duel",
        configuration_sha256="a" * 64,
        model_dir=model_dir,
    )
    assert resumed.completed_timesteps == 2
    assert resumed.completed_updates == 1
    assert [model.num_timesteps for model in resumed.models.values()] == [2, 2]


def test_resume_restores_seed_for_the_next_checkpoint(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()

    resumed = SelfPlayTrainer.from_checkpoint(
        FakeEnv(),
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
        checkpoint,
        run_name="duel",
        configuration_sha256="a" * 64,
        seed=0,
        model_dir=model_dir,
    )
    assert resumed.seed == 17
    resumed.learn(4)
    next_checkpoint = resumed.save_checkpoint()
    assert json.loads((next_checkpoint / "state.json").read_text())["seed"] == 17


def test_resume_rejects_changed_configuration_hash(tmp_path):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()

    with pytest.raises(ValueError, match="configuration_sha256"):
        trainer.load_checkpoint(checkpoint, configuration_sha256="b" * 64)


def test_resume_rejects_one_missing_model(tmp_path):
    trainer = make_trainer(
        model_dir=tmp_path / "models" / "duel",
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()
    (checkpoint / "player_1.zip").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        trainer.load_checkpoint(checkpoint)


def test_synthetic_keyboard_interrupt_does_not_publish_unsafe_pair(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )

    def interrupted_train():
        raise KeyboardInterrupt

    trainer.models["player_1"].train = interrupted_train
    with pytest.raises(KeyboardInterrupt):
        trainer.learn(2)
    assert not model_dir.exists()


def test_deferred_sigint_publishes_matching_update_markers(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    for model in trainer.models.values():
        model.update_marker = 0
    calls = []
    original_player_0_train = trainer.models["player_0"].train
    original_player_1_train = trainer.models["player_1"].train
    prior_handler = object()
    installed_handlers = []

    def fake_getsignal(signum):
        assert signum == signal.SIGINT
        return prior_handler

    def fake_signal(signum, handler):
        assert signum == signal.SIGINT
        installed_handlers.append(handler)
        return prior_handler

    monkeypatch.setattr(self_play_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(self_play_module.signal, "signal", fake_signal)
    original_save_checkpoint = trainer.save_checkpoint

    def save_checkpoint(*args, **kwargs):
        assert installed_handlers[-1] is not prior_handler
        assert trainer.completed_timesteps == 2
        assert trainer.completed_updates == 1
        return original_save_checkpoint(*args, **kwargs)

    trainer.save_checkpoint = save_checkpoint

    def train_player_0():
        calls.append("player_0")
        trainer.models["player_0"].update_marker += 1
        installed_handlers[0](signal.SIGINT, None)
        return original_player_0_train()

    def train_player_1():
        calls.append("player_1")
        trainer.models["player_1"].update_marker += 1
        return original_player_1_train()

    trainer.models["player_0"].train = train_player_0
    trainer.models["player_1"].train = train_player_1
    created = []
    original_mkdtemp = self_play_module.tempfile.mkdtemp

    def record_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(self_play_module.tempfile, "mkdtemp", record_mkdtemp)
    with pytest.raises(KeyboardInterrupt):
        trainer.learn(2)

    checkpoint = model_dir / "checkpoints" / "000000000002"
    saved = {agent_id: PPO.load(str(checkpoint / f"{agent_id}.zip")) for agent_id in trainer.agent_ids}
    assert calls == ["player_0", "player_1"]
    assert [saved[agent_id].update_marker for agent_id in trainer.agent_ids] == [1, 1]
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["completed_timesteps"] == 2
    assert state["completed_updates"] == 1
    assert installed_handlers[-1] is prior_handler
    assert len(created) == 1
    assert created[0].startswith(str(model_dir / "checkpoints"))


def test_worker_train_keyboard_interrupt_is_unsafe_without_checkpoint(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    calls = []

    def train_player_0():
        calls.append("player_0")

    def interrupt_player_1():
        calls.append("player_1")
        raise KeyboardInterrupt

    trainer.models["player_0"].train = train_player_0
    trainer.models["player_1"].train = interrupt_player_1
    errors = []

    def run_training():
        try:
            trainer.learn(2)
        except KeyboardInterrupt as error:
            errors.append(error)

    worker = threading.Thread(target=run_training)
    worker.start()
    worker.join()
    assert calls == ["player_0", "player_1"]
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert trainer._unsafe_update_interruption is True
    assert not model_dir.exists()


def test_interrupted_second_model_save_cleans_staging_without_publishing(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    trainer = make_trainer(
        model_dir=model_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )

    def interrupt_save(path):
        raise KeyboardInterrupt

    trainer.models["player_1"].save = interrupt_save
    with pytest.raises(KeyboardInterrupt):
        trainer.save_checkpoint()
    checkpoints = model_dir / "checkpoints"
    assert not (model_dir / "latest.json").exists()
    assert not (checkpoints / "000000000000").exists()
    assert list(checkpoints.iterdir()) == []


def test_interrupted_latest_write_cleans_temporary_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "duel"
    checkpoint = model_dir / "checkpoints" / "000000000000"
    checkpoint.mkdir(parents=True)

    def interrupt_replace(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(self_play_module.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        SelfPlayTrainer._write_latest(model_dir, checkpoint)
    assert list(model_dir.glob(".latest-*")) == []
    assert not (model_dir / "latest.json").exists()


def test_metrics_append_after_resume(tmp_path):
    model_dir = tmp_path / "models" / "duel"
    run_dir = tmp_path / "runs" / "duel"
    trainer = make_trainer(
        truncated=True,
        model_dir=model_dir,
        run_dir=run_dir,
        run_name="duel",
        configuration_sha256="a" * 64,
    )
    trainer.learn(2)
    checkpoint = trainer.save_checkpoint()
    first = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(first) == 1
    assert json.loads(first[0])["player_0"]["mean_reward"] == 1.0

    resumed = SelfPlayTrainer.from_checkpoint(
        FakeEnv(truncated=True),
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
        checkpoint,
        run_name="duel",
        configuration_sha256="a" * 64,
        model_dir=model_dir,
        run_dir=run_dir,
    )
    resumed.learn(4)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["completed_timesteps"] for record in records] == [2, 4]
    assert len(records) == 2
