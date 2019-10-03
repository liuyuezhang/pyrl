from envs.atari.wrappers import *
import gym


def make_env(env_id, seed=0, stack_frames=1, max_episode_steps=2500, episodic_life=True, reward_clipping=True):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env.seed(seed)
    env._max_episode_steps = max_episode_steps * 4  # do not change the position
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if episodic_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)

    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)
    env = ScaledFloatFrame(env)
    return env
