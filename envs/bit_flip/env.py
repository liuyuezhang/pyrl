from envs.bit_flip.bit_flip import BitFlip


def make_env(n=10):
    return BitFlip(n)
