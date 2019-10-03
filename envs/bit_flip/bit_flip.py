from gym import spaces

import numpy as np


class BitFlip:
    def __init__(self, n):
        self.n = n

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n, ), dtype=np.uint8)
        self.action_space = spaces.Discrete(n + 1)  # 0 to n-1, n: no no-op

        self.state = self.observation_space.sample()
        self.target = self.observation_space.sample()
        self.t = 0

    def reset(self):
        self.state = self.observation_space.sample()
        self.target = self.observation_space.sample()
        self.t = 0
        return np.copy(self.state), np.copy(self.target)

    def step(self, action):
        assert 0 <= action <= self.n
        if action == self.n:  # no-op
            pass
        else:
            self.state[action] = 1.0 - self.state[action]
        self.t += 1

        dist = np.sum(np.bitwise_xor(self.state, self.target))
        reward = 1.0 if dist == 0 else 0.0
        done = True if dist == 0 or self.t >= self.n else False

        return (np.copy(self.state), np.copy(self.target)), reward, done, None

    def seed(self, seed):
        np.random.seed(seed)
