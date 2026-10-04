"""Real engine tests. Set GODOT_RL_ADDON to the separate add-on checkout."""

import os
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from godot_rl import project_cli as cli
from godot_rl.core.multi_agent_env import GodotMultiAgentEnv
from godot_rl.core.project_env import GodotProjectEnv
from godot_rl.core.protocol import encode_frame, recv_frame, send_frame
from godot_rl.training.self_play import SelfPlayTrainer
from godot_rl.wrappers.project_sb3 import GodotProjectVecEnv


@pytest.fixture
def game(tmp_path):
    addon = os.environ.get("GODOT_RL_ADDON")
    godot = os.environ.get("GODOT_BIN") or shutil.which("godot") or shutil.which("godot4")
    if not godot and Path("/Applications/Godot.app/Contents/MacOS/Godot").is_file():
        godot = "/Applications/Godot.app/Contents/MacOS/Godot"
    if not addon or not godot:
        pytest.skip("Set GODOT_RL_ADDON and install Godot to run engine tests")
    project = tmp_path / "game"
    shutil.copytree(Path(__file__).parent / "fixtures/project_protocol", project)
    shutil.copytree(
        Path(addon) / "addons/godot_rl_agents",
        project / "addons/godot_rl_agents",
        ignore=shutil.ignore_patterns("*.uid", "*.import"),
    )
    imported = subprocess.run(
        [godot, "--headless", "--editor", "--path", str(project), "--quit"], capture_output=True, text=True, timeout=30
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert "SCRIPT ERROR" not in imported.stderr, imported.stderr
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return {"godot_path": Path(godot), "project_path": project, "port": port, "speedup": 8.0}


def test_real_single_agent_checker_and_episode_flags(game):
    env = GodotProjectEnv(scene="res://rl_training.tscn", **game)
    try:
        check_env(env, skip_render_check=True)
        for action, expected in [(-0.5, (True, False)), (0.5, (False, True))]:
            env.reset(seed=17)
            for _ in range(12):
                observation, _, terminated, truncated, info = env.step({"action": np.array([action], dtype=np.float32)})
                if terminated or truncated:
                    assert (terminated, truncated) == expected
                    np.testing.assert_allclose(info["terminal_observation"]["obs"], observation["obs"])
                    break
            else:
                pytest.fail("The fixture did not finish an episode")
    finally:
        env.close()
    assert env.process.poll() is not None


def test_real_legacy_client_accepts_fragmented_and_combined_frames(game):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", game["port"]))
        listener.listen(1)
        listener.settimeout(10)
        process = subprocess.Popen(
            [
                str(game["godot_path"]),
                "--headless",
                "--path",
                str(game["project_path"]),
                "--scene",
                "res://rl_training.tscn",
                f'--port={game["port"]}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(10)
                frames = encode_frame({"type": "handshake", "major_version": "0", "minor_version": "7"})
                frames += encode_frame({"type": "env_info"})
                for offset in range(0, len(frames), 3):
                    connection.sendall(frames[offset : offset + 3])
                info = recv_frame(connection)
                assert info["n_agents"] == 1
                assert isinstance(info["observation_space"], list)
                send_frame(connection, {"type": "action", "action": [{"action": [-0.5]}]})
                step = recv_frame(connection)
                assert isinstance(step["obs"], list)
                assert isinstance(step["done"], list)
                assert isinstance(step["terminated"], list)
                send_frame(connection, {"type": "close"})
            process.wait(timeout=5)
            assert process.returncode == 0, process.stderr.read().decode()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()


def test_real_self_play_checkpoint_resume_and_evaluation(game, capsys):
    project = game["project_path"]
    cli.init_project(project, "res://self_play_training.tscn", "self-play", sync=False)
    configuration = (project / "rl/config.toml").read_text()
    configuration = configuration.replace("n_steps = 128", "n_steps = 8")
    configuration = configuration.replace("batch_size = 64", "batch_size = 8")
    configuration = configuration.replace("n_epochs = 10", "n_epochs = 1")
    configuration = configuration.replace("checkpoint_interval = 8192", "checkpoint_interval = 8")
    configuration = configuration.replace("port = 11008", f'port = {game["port"]}')
    (project / "rl/config.toml").write_text(configuration)
    godot = str(game["godot_path"])
    assert cli.doctor(project, godot)["ok"]
    assert cli.validate_project(project, 16, godot) == 0
    assert cli.self_play_project(project, 16, "duel", godot=godot) == 0
    checkpoint = project / "rl/models/duel/checkpoints/000000000016"
    assert {path.name for path in checkpoint.iterdir()} == {"player_0.zip", "player_1.zip", "state.json"}
    assert cli.self_play_project(project, 24, "duel", resume=str(checkpoint), godot=godot) == 0
    checkpoint = project / "rl/models/duel/checkpoints/000000000024"
    before = {path: path.read_bytes() for path in (project / "rl/models").rglob("*") if path.is_file()}
    capsys.readouterr()
    assert cli.evaluate_project(project, str(checkpoint), 4, max_steps=16, godot=godot) == 0
    after = {path: path.read_bytes() for path in (project / "rl/models").rglob("*") if path.is_file()}
    assert before == after


def test_real_single_agent_ppo_resets_episodes(game):
    raw = GodotProjectEnv(scene="res://rl_training.tscn", **game)
    vec = GodotProjectVecEnv(raw)
    try:
        model = PPO("MultiInputPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, device="cpu")
        model.learn(16)
        assert model.num_timesteps == 16
    finally:
        vec.close()
    assert raw.process.poll() is not None


def test_real_two_agent_synchronized_rollouts(game):
    env = GodotMultiAgentEnv(scene="res://self_play_training.tscn", **game)
    try:
        env.reset(seed=17)
        for _ in range(16):
            observations, rewards, terminated, truncated, infos = env.step(
                {"player_0": {"action": 0}, "player_1": {"action": 1}}
            )
            if any(terminated.values()):
                assert all(terminated.values())
                assert not any(truncated.values())
                assert rewards == {"player_0": 1.0, "player_1": -1.0}
                assert [infos[key]["outcome"] for key in env.agent_ids] == ["win", "loss"]
                break
        else:
            pytest.fail("The two-agent fixture did not finish")
        trainer = SelfPlayTrainer(env, {"n_steps": 8, "batch_size": 8, "n_epochs": 1}, seed=17)
        trainer.learn(16)
        assert [model.num_timesteps for model in trainer.models.values()] == [16, 16]
    finally:
        env.close()
    assert env.process.poll() is not None
