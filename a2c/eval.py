from envs.atari.env import make_env
from a2c.model import AC_LSTM
from common.logger import Logger

import os
import time
import argparse
import cv2
import numpy as np

import torch
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A3C_LSTM_EVAL')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--max-episode-steps', type=int, default=10000)
    parser.add_argument('--stacked-frames', type=int, default=1)
    parser.add_argument('--render', default=True)
    parser.add_argument('--fps', type=int, default=500)
    parser.add_argument('--cuda', type=int, default=True)
    args = parser.parse_args()

    path = './res/' + args.env + '_' + 'a3c-lstm' + '_' + str(args.seed) + '/'
    assert os.path.exists(path)

    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    env = make_env(args.env, seed=args.seed, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=False)

    net = AC_LSTM(env.observation_space.shape[0], env.action_space.n, debug=True).to(device)
    saved_state = torch.load(path + 'model.dat')
    net.load_state_dict(saved_state)
    net.eval()

    logger = Logger(name="eval", path=path, print_log=True, save_model=False)
    logger.add_model(net)

    for i in range(args.num_episodes):
        state = env.reset()
        state_v = torch.from_numpy(state).float().to(device)
        hx = torch.zeros(1, 512).to(device)
        cx = torch.zeros(1, 512).to(device)

        t = 0
        while True:
            # Perform action according to policy
            with torch.no_grad():
                value_v, logit_v, (hx, cx), conv3 = net(state_v.unsqueeze(0), (hx, cx))
            prob_v = F.softmax(logit_v, dim=1)
            action_v = torch.multinomial(prob_v, num_samples=1)
            action = int(action_v.item())

            if args.render:
                env.render()
                time.sleep(1.0 / args.fps)

                conv3 = np.mean(conv3.squeeze(), axis=0)
                cv2.imshow('Frame', conv3)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Receive reward and new state
            state, reward, done, info = env.step(action)
            state_v = torch.from_numpy(state).float().to(device)

            t += 1
            print(t, done, info)
            logger.log(t, reward, info)

            if done and not info:
                state = env.reset()
                state_v = torch.from_numpy(state).float().to(device)
                hx = torch.zeros(1, 512).to(device)
                cx = torch.zeros(1, 512).to(device)

            if info:
                break
