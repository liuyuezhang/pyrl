import argparse
import os

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO-LSTM')
    parser.add_argument('--env',               type=str,   default="LunarLander-v2")
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--num-episodes',      type=int,   default=1000)
    parser.add_argument('--num-steps',         type=int,   default=300)
    parser.add_argument('--num-processes',     type=int,   default=8)

    parser.add_argument('--lr',                type=float, default=2e-3)
    parser.add_argument('--gamma',             type=float, default=0.99)
    parser.add_argument('--T',                 type=int,   default=200)
    parser.add_argument('--K_epochs',          type=int,   default=4)
    parser.add_argument('--eps_clip',          type=float, default=0.2)

    parser.add_argument('--cuda',                          default=False)
    args = parser.parse_args()

    N = args.num_processes
    T = args.T

    # seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # path
    path = './res/' + args.env + '_' + 'ppo' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # device
    device = torch.device("cuda" if args.cuda else "cpu")

    # env: vectorized environment, auto reset
    from envs.gym.env import make_env
    from common.vec_env.subproc_vec_env import SubprocVecEnv
    env_fns = []
    for idx in range(args.num_processes):
        env_fns.append(lambda: make_env(args.env, seed=args.seed + idx))
    venv = SubprocVecEnv(env_fns)
    state_dim = venv.observation_space.shape[0]
    action_dim = venv.action_space.n

    # agent
    from ppo.agent import PPO
    agent = PPO(N=N, T=T, state_dim=state_dim, action_dim=action_dim,
                lr=args.lr, gamma=args.gamma, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
                device=device)
    
    # logger
    from common.vec_env.vec_logger import VecLogger
    logger = VecLogger(path=path, N=N, print_freq=100)
    # logger.add_model(agent.policy)
    
    # training loop
    t = 0
    cnt = 0
    state = venv.reset()
    for e in range(args.num_episodes):
        for _ in range(args.num_steps):
            # perform action
            action = agent.act(state)
            state, reward, done, info = venv.step(action)

            t += N
            cnt += 1
            logger.log(t, reward, done)

            # restore reward and done
            agent.step(reward, done)

            # update
            if cnt % T == 0:
                agent.update()
                cnt = 0
