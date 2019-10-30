import gym


def make_env(env_id, seed=0, max_episode_steps=2500):
    env = gym.make(env_id)
    env.seed(seed)
    env._max_episode_steps = max_episode_steps
    return env
