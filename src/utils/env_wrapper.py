import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Discrete


class ObsArrayWrapper(ObservationWrapper):
    def observation(self, obs: int | float | np.integer | np.floating | np.ndarray):
        # Only wrap scalars into a 1D array
        if isinstance(obs, (int, float, np.integer, np.floating)):
            return np.array(obs, dtype=np.float32)

        return obs  # leave arrays as-is


class SafeActionWrapper(ActionWrapper):
    def action(self, action):
        if isinstance(self.action_space, Discrete):
            return int(action)  # discrete → Python int

        return action
