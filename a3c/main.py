from envs.atari.env import make_env
from a3c.model import AC_LSTM
from a3c.train import train
from a3c.test import test
from common.shared_optim import SharedRMSprop, SharedAdam

import os
import argparse

import torch
import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"  # critical for high FPS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A3C-LSTM')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=20e6)
    parser.add_argument('--max-episode-steps', type=int, default=2500)
    parser.add_argument('--stacked-frames', type=int, default=1)
    parser.add_argument('--num-processes', type=int, default=16)
    parser.add_argument('--cuda', default=False)

    parser.add_argument('--shared-optimizer', default=True)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--amsgrad', default=True)
    args = parser.parse_args()

    path = './res/' + args.env + '_' + 'a3c-lstm' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # Assume global shared parameter vectors and global shared counter T
    env = make_env(args.env, stack_frames=args.stacked_frames)
    shared_net = AC_LSTM(env.observation_space.shape[0], env.action_space.n)
    shared_net.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_net.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_net.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    T = mp.Value('i', 0)
    lock = mp.Lock()
    processes = []

    # Hogwild! training
    p = mp.Process(target=test, args=(args.num_processes, args, T, shared_net, path))
    p.start()
    processes.append(p)
    for idx in range(args.num_processes):
        p = mp.Process(target=train, args=(idx, args, T, lock, shared_net, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
