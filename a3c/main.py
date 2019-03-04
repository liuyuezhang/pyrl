from a3c.env import make_env
from a3c.model import A3C
from a3c.train import train
from a3c.test import test
from a3c.shared_optim import SharedRMSprop, SharedAdam

import os
import argparse

import torch
import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"  # critical for high FPS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=50e6)
    parser.add_argument('--max_episode_steps', type=int, default=10000)
    parser.add_argument('--stacked_frames', type=int, default=4)
    parser.add_argument('--num_processes', type=int, default=32)
    parser.add_argument('--cuda', default=True)

    parser.add_argument('--shared-optimizer', default=True)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--amsgrad', default=True)

    args = parser.parse_args()

    path = './res/' + args.env + '_' + 'a3c' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    mp.set_start_method('spawn')  # required by CUDA, and PyTorch 0.4.1 has a bug of mp in spawn mode

    # Assume global shared parameter vectors and global shared counter T
    env = make_env(args.env)
    shared_net = A3C(env.observation_space.shape[0], env.action_space)
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
    processes = []

    # Hogwild! training
    p = mp.Process(target=test, args=(args.num_processes, args, T, shared_net, path))
    p.start()
    processes.append(p)
    for idx in range(args.num_processes):
        p = mp.Process(target=train, args=(idx, args, T, shared_net, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
