from envs.atari.env import make_env
from a3c.model import AtariCnnAcLstm
from common.logger import Logger

import torch
import torch.nn.functional as F


def test(args, T, shared_net, path):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    env = make_env(args.env, seed=args.seed, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=False)

    net = AtariCnnAcLstm(env.observation_space.shape[0], env.action_space.n)
    net.eval()

    logger = Logger(name="test", path=path, print_log=True, save_model=True)
    logger.add_model(model=net)

    state = env.reset()
    state_v = torch.from_numpy(state).float()
    hx = torch.zeros(1, 512)
    cx = torch.zeros(1, 512)
    info = True  # game is real done, not end of a life (EpisodicLife)

    while T.value < args.num_timesteps:
        # Synchronize thread-specific parameters
        if info:
            net.load_state_dict(shared_net.state_dict())

        # Perform action according to policy
        with torch.no_grad():
            value_v, logit_v, (hx, cx) = shared_net(state_v.unsqueeze(0), (hx, cx))
        prob_v = F.softmax(logit_v, dim=1)
        action_v = torch.multinomial(prob_v, num_samples=1)
        action = int(action_v.item())

        # Receive reward and new state
        state, reward, done, info = env.step(action)
        logger.log(T.value, reward, info)

        state_v = torch.from_numpy(state).float()

        if done:
            state = env.reset()
            state_v = torch.from_numpy(state).float()
            hx = torch.zeros(1, 512)
            cx = torch.zeros(1, 512)
