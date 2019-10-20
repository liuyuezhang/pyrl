import argparse
import os

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO-LSTM')
    parser.add_argument('--env',               type=str,   default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed',              type=int,   default=4)
    parser.add_argument('--num-timesteps',     type=int,   default=20e6)
    parser.add_argument('--max-episode-steps', type=int,   default=2500)
    parser.add_argument('--stacked-frames',    type=int,   default=1)
    parser.add_argument('--num-processes',     type=int,   default=16)
    parser.add_argument('--cuda',                          default=True)

    parser.add_argument('--optimizer',         type=str,   default='Adam')
    parser.add_argument('--lr',                type=float, default=2.5e-4)
    parser.add_argument('--amsgrad',                       default=True)
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # path
    path = './res/' + args.env + '_' + 'ppo-lstm' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # device
    device = torch.device("cuda" if args.cuda else "cpu")

    # env: vectorization, automatic reset() in SubprocVecEnv
    from envs.atari.env import make_env
    from common.vec_env.subproc_vec_env import SubprocVecEnv
    env_fns = []
    for idx in range(1, args.num_processes + 1):
        env_fns.append(lambda: make_env(args.env, seed=args.seed + idx))
    venv = SubprocVecEnv(env_fns)

    # model
    from ppo.model import AtariCnnAcLstm
    model = AtariCnnAcLstm

    # train
    from ppo.train import train
    train(args, venv=venv, model=model, path=path, device=device)
