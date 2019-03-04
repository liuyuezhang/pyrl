from a3c.env import make_env
from a3c.model import A3C
from common.logger import Logger

import os
import time
import argparse

import torch
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A3C_EVAL')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--max_episode_steps', type=int, default=2500)
    parser.add_argument('--stacked_frames', type=int, default=4)
    parser.add_argument('--render', default=True)
    parser.add_argument('--render-freq', type=int, default=1)
    parser.add_argument('--fps', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=True)
    args = parser.parse_args()

    path = './res/' + args.env + '_' + 'a3c' + '_' + str(args.seed) + '/'
    assert os.path.exists(path)

    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    env = make_env(args.env, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=False)
    env.seed(args.seed)

    net = A3C(env.observation_space.shape[0], env.action_space).to(device)
    saved_state = torch.load(path + 'model.dat', map_location=lambda storage, loc: storage)
    net.load_state_dict(saved_state)
    net.eval()

    logger = Logger(name="eval", path=path, model=net, start_time=time.time(),
                    print_log=True, save_model=False)

    for i in range(args.num_episodes):
        state = env.reset()
        state_v = torch.from_numpy(state).float().to(device)

        t = 0
        while True:
            if args.render and i % args.render_freq == 0:
                env.render()
                if args.fps > 0:
                    time.sleep(1.0 / args.fps)

            # Perform action according to policy
            with torch.no_grad():
                value_v, logit_v = net(state_v.unsqueeze(0))
            prob_v = F.softmax(logit_v, dim=1)
            # action_v = torch.argmax(prob_v, dim=1)
            action_v = torch.multinomial(prob_v, num_samples=1)
            action = int(action_v.item())

            # Receive reward and new state
            state, reward, done, info = env.step(action)
            state_v = torch.from_numpy(state).float().to(device)

            t += 1
            print(t, done, info)
            logger.log(t, reward, info)

            if done and not info:
                state = env.reset()
                state_v = torch.from_numpy(state).float().to(device)

            if info:
                break
