from envs.fetch.env import make_env

env = make_env('FetchReach-v1')
env.seed(0)

while True:
    ob = env.reset()
    done = False
    print(ob)
    while not done:
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        print(ob, reward, done, info)
