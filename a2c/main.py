from envs.atari.env import make_env
from a2c.model import AC_LSTM
from a2c.train import train

from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.vec_frame_stack import VecFrameStack

import os
import argparse

import torch
import torch.optim as optim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C LSTM')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=20e6)
    parser.add_argument('--max-episode-steps', type=int, default=2500)
    parser.add_argument('--stacked-frames', type=int, default=1)
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--cuda', default=True)

    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--amsgrad', default=True)
    args = parser.parse_args()

    path = './res/' + args.env + '_' + 'a2c-lstm' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # env
    env_fns = []
    for idx in range(args.num_processes):
        env_fns.append(lambda: make_env(args.env, seed=args.seed + idx))
    venv = SubprocVecEnv(env_fns)

    # policy
    net = AC_LSTM(venv.observation_space.shape[0], venv.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # train(args, env=venv, logger, optimizer, device)
