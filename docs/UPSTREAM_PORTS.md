# Contributions from the toolkit

The Python fork keeps the upstream package name, examples, wrappers, and documentation.
The add-on stays in its separate repository.
The port excludes fork branding, namespace changes, cleanup deletions, the fork upgrade command, and bundled add-on assets.

Each feature branch adds one feature commit above its base.
Dependent branches form a stack, which means that each branch starts from the previous contribution.
For review, compare each branch with the base in this table.
For an upstream PR, wait for its dependencies to merge, then rebase the feature commit onto upstream `main`.
An immediate PR against upstream `main` includes any dependencies that upstream does not contain yet.
Both fork `main` branches contain the contributions, and the feature branches remain available.

## Python branches

| Branch | Review base | Change |
| --- | --- | --- |
| `sb3-episode-handling` | `sb3-vecenv` | Preserve final observations and support asynchronous single-observation training. |
| `doctor-app-bundles` | `doctor` | Recognize the executable inside a macOS app bundle. |
| `project-environment` | `protocol-two` | Launch a Godot project as a single-agent Gymnasium environment. |
| `project-sb3` | `project-environment` | Adapt the project environment to SB3 and reset completed episodes. |
| `two-agent-environment` | `project-sb3` | Exchange actions and results for two players in one game. |
| `synchronized-ppo` | `two-agent-environment` | Collect shared steps before updating either policy. |
| `paired-checkpoints` | `synchronized-ppo` | Save both policies together, resume them, and append metrics. |
| `project-commands` | `paired-checkpoints` | Add project setup, diagnostics, validation, and single-agent training. |
| `self-play-commands` | `project-commands` | Add self-play training and deterministic evaluation with swapped player seats. |
| `real-godot-coverage` | `self-play-commands` | Exercise both protocols, training, resume, and evaluation in Godot. |
| `upstream-compatibility` | `real-godot-coverage` | Test SB3 2.4.0 and 2.9.0 and apply repository formatting. |
| `contribution-guide` | Fork `main` after the feature merges | Document the series and test command dispatch. |

The existing `protocol-safety`, `protocol-two`, `gymnasium-env`, `one-agent-smoke`, `sb3-vecenv`, and `doctor` branches remain separate.
The `protocol-safety` branch still contains its single squashed commit.
The redundant `gymnasium-termination` branch stays absent from GitHub.

## Godot add-on branches

The add-on fork is `MyNameIsArko/godot_rl_agents_plugin`.

| Branch | Review base | Change |
| --- | --- | --- |
| `episode-signals` | Upstream `main` | Add termination, truncation, and cached final observations to 2D and 3D controllers. |
| `framed-transport` | `episode-signals` | Buffer partial messages, limit message size, and reject invalid values. |
| `project-protocol` | `framed-transport` | Negotiate protocol 1 while retaining the legacy 0.7 protocol. |
| `stable-player-identities` | `project-protocol` | Add protocol 2 and stable `player_0` and `player_1` identities. |

The add-on preserves ONNX inference, sensors, rewards, policy names, and demonstration recording.
Existing `done` assignments still work.
The new `terminated` and `truncated` fields provide separate episode signals.
Protocol 2 requires two agents with matching spaces, shared episode endings, and complementary outcomes.

## Project setup

Use the Python fork as the package source during development.
Pass the separate add-on checkout to copy the add-on into a game.
The command refuses changes to existing managed files before it writes any file.

```sh
gdrl init --project /path/to/game --scene res://rl_training.tscn \
  --addon-path /path/to/godot_rl_agents_plugin \
  --package-source /path/to/godot_rl_agents
gdrl doctor --project /path/to/game
gdrl validate --project /path/to/game --steps 32
gdrl train --project /path/to/game --timesteps 1024 --name first-run
```

For self-play, add `--mode self-play` during initialization.
Set the two controller identifiers to `player_0` and `player_1`.
Implement shared reset and episode outcomes in the game.
Use `gdrl self-play` to train both policies and `gdrl evaluate` to evaluate a paired checkpoint.
Evaluation uses deterministic actions and swaps the two policies between player seats.

Use `--no-sync` to create project files without installing dependencies.
Doctor uses the project virtual environment when it exists, or the current Python interpreter otherwise.
Doctor reports the scene, add-on capabilities, Godot executable and version, Python imports, and port availability.
`gdrl doctor --env_path ...` keeps the earlier executable diagnostics.

Resume restores both policies, their optimizers, and shared training counters.
Resume starts a new game episode.
It does not restore game state or promise the same random sequence as uninterrupted training.

## Validation

Local tests pass with Gymnasium 1.0.0, SB3 2.4.0, and PyTorch 2.8.0.
The same contribution tests also run with Gymnasium 1.3.0 and SB3 2.9.0.
Real engine tests use Godot 4.7.2 on macOS.
The real engine tests cover final observations, episode resets, legacy framing, two-policy training, paired saves, resume, and evaluation.
The downloaded upstream example games, ONNX inference, demonstration recording, and other trainers need their separate integration tests.

From the Python checkout, run the engine tests with the compatible add-on checkout available:

```sh
GODOT_RL_ADDON=/path/to/godot_rl_agents_plugin \
GODOT_BIN=/path/to/godot \
python -m pytest -q tests/test_real_godot.py
```

The Python wheel does not contain the add-on.
Coordinate upstream Python and add-on PRs before releasing protocols 1 and 2.
