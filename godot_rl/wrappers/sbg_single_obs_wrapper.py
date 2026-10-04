from typing import Any, Dict, List, Tuple

import numpy as np

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# A variant of the Stable Baselines Godot Env that only supports a single obs space from the dictionary - obs["obs"] by default.
# This provides some basic support for using envs that have a single obs space with policies other than MultiInputPolicy.


class SBGSingleObsEnv(StableBaselinesGodotEnv):
    def __init__(self, obs_key="obs", *args, **kwargs) -> None:
        self.obs_key = obs_key
        super().__init__(*args, **kwargs)
        self.observation_space = self.envs[0].observation_space[self.obs_key]

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs, rewards, term, info = super().step_wait()

        # Terminal obs info is needed for imitation learning
        for idx, done in enumerate(term):
            if done:
                terminal = info[idx]["terminal_observation"]
                if isinstance(terminal, dict):
                    info[idx]["terminal_observation"] = terminal[self.obs_key]

        return obs[self.obs_key], rewards, term, info

    def reset(self) -> np.ndarray:
        obs = super().reset()
        return obs[self.obs_key]
