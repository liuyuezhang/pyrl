import argparse
import os
import numpy as np

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Breakout')
    parser.add_argument('--max-episode-steps', type=int, default=10000)
    parser.add_argument('--stack-frames', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--total-steps',   type=int,   default=int(20e6))
    parser.add_argument('--init-steps',    type=int,   default=50000)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--buffer-size',   type=int,   default=int(1e6))
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--target-freq',   type=int,   default=10000)
    parser.add_argument('--update-freq',   type=int,   default=4)
    parser.add_argument('--update-steps',  type=int,   default=1)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-final', type=float, default=0.1)
    parser.add_argument('--epsilon-steps', type=int,   default=int(1e6))

    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # path
    path = './res/' + args.env + '_' + 'ddqn' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # device
    device = torch.device("cuda" if args.cuda else "cpu")

    # env
    from envs.atari.env import make_env
    env = make_env(args.env + 'NoFrameskip-v4', seed=args.seed, stack_frames=args.stack_frames,
                   max_episode_steps=args.max_episode_steps, episodic_life=True, reward_clipping=True)

    # model
    from ddqn.model import DQN
    model = DQN

    # train
    from ddqn.train import train
    train(args, env=env, model=model, path=path, device=device)
