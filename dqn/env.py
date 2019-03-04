from envs.ale.wrappers import *


def make_env(env, seed, repeat_act):
    env = ALE(env, seed, repeat_act)
    env = EpisodicLife(env)
    env = NoopReset(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = FireReset(env)
    env = ProcessFrame84(env)
    env = StackedFrame(env, stack=4, dtype=np.uint8)
    env = ImageToPyTorch(env)
    return env
