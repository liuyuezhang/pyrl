import gym
import numpy as np


def make_env(env_id):
    env = gym.make(env_id)
    env = MyFetchEnv(env)
    return env


class MyFetchEnv(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env):
        super(MyFetchEnv, self).__init__(env)

        # observation space
        self.ob_shape = self.env.observation_space.spaces['observation'].shape
        self.goal_shape = self.env.observation_space.spaces['achieved_goal'].shape
        size = np.prod(self.ob_shape) + np.prod(self.goal_shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        ob = np.concatenate([state['achieved_goal'].ravel(), state['observation'].ravel()])
        return ob, reward, done, info

    def reset(self):
        state = self.env.reset()
        ob = np.concatenate([state['achieved_goal'].ravel(), state['observation'].ravel()])
        target = state['desired_goal']
        return ob, target
