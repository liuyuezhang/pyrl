from envs.atari.env import make_env
from common.logger import Logger

import torch
import torch.nn.functional as F


def test(args, T, shared_net, path):
    device = next(shared_net.parameters()).device

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    env = make_env(args.env, seed=args.seed, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=False)

    state = env.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(1, 512).to(device)
    cx = torch.zeros(1, 512).to(device)

    logger = Logger(name="test", path=path, print_log=True, save_model=True)
    logger.add_model(model=shared_net)

    while T.value < args.num_timesteps:
        with torch.no_grad():
            # Perform action according to policy
            value_v, logit_v, (hx, cx) = shared_net(state_v.unsqueeze(0), (hx, cx))
            prob_v = F.softmax(logit_v, dim=1)
            action_v = torch.multinomial(prob_v, num_samples=1)
            action = int(action_v.item())

            # Receive reward and new state
            state, reward, done, info = env.step(action)
            state_v = torch.from_numpy(state).float().to(device)

            logger.log(T.value, reward, info)

            if done:
                state = env.reset()
                state_v = torch.from_numpy(state).float().to(device)
                hx = torch.zeros(1, 512).to(device)
                cx = torch.zeros(1, 512).to(device)
