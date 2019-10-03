from envs.bit_flip.env import make_env

env = make_env(2)
env.seed(0)

while True:
    ob = env.reset()
    done = False
    print(ob)
    while not done:
        action = int(input())
        ob, reward, done, info = env.step(action)
        success = 1 if reward == 0 else 0
        print(ob, reward, done, success)
