from envs.atari.env import make_env
from a2c.model import AC_LSTM
from a2c.train import train
from a2c.test import test

import os
import argparse

import torch
import torch.multiprocessing as mp
import torch.optim as optim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C LSTM')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=20e6)
    parser.add_argument('--max-episode-steps', type=int, default=2500)
    parser.add_argument('--stacked-frames', type=int, default=1)
    parser.add_argument('--num-processes', type=int, default=16)
    parser.add_argument('--cuda', default=False)

    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--amsgrad', default=True)
    args = parser.parse_args()

    # path
    path = './res/' + args.env + '_' + 'a2c-lstm' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # policy
    device = torch.device("cuda" if args.cuda else "cpu")
    env = make_env(args.env, stack_frames=args.stacked_frames)
    net = AC_LSTM(env.observation_space.shape[0], env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    # training and testing
    T = mp.Value('i', 0)
    lock = mp.Lock()

    p1 = mp.Process(target=test, args=(args, T, net, path))
    p1.start()
    p2 = mp.Process(target=train, args=(args, T, lock, net, optimizer))
    p2.start()
    p1.join()
    p2.join()
