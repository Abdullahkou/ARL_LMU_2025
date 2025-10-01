import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete


class ObsArrayWrapper(ObservationWrapper):
    def observation(self, obs: int | float | np.integer | np.floating | np.ndarray):
        # Only wrap scalars into a 1D array
        if isinstance(obs, (int, float, np.integer, np.floating)):
            return np.array(obs)

        return obs  # leave arrays as-is


class AutoOneHotWrapper(ObservationWrapper):
    """
    - For Discrete(n) obs spaces: returns one-hot encoded vector of length n
    - For Box spaces: passes obs through unchanged
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_space = env.observation_space

        if isinstance(env.observation_space, Discrete):
            n = env.observation_space.n
            self.observation_space = Box(0, 1, shape=(n,))
            self._mode = "onehot"
            self._n = n
        elif isinstance(env.observation_space, Box):
            self.observation_space = env.observation_space
            self._mode = "box"
        else:
            raise NotImplementedError(
                "AutoOneHotWrapper only supports Discrete and Box obs spaces."
            )

    def observation(self, obs: int | float | np.integer | np.floating | np.ndarray):
        if self._mode == "onehot":
            onehot = np.zeros(self._n)
            onehot[obs] = 1.0

            return onehot

        return obs


class SafeActionWrapper(ActionWrapper):
    def action(self, action):
        if isinstance(self.action_space, Discrete):
            return int(action)  # discrete → Python int

        return action
