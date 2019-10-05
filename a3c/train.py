from envs.atari.env import make_env
from a3c.model import AC_LSTM

import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
TAU = 1.00
REWARD_STEPS = 20
CLIP_GRAD = 50

COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(idx, args, T, lock, shared_net, optimizer):
    if args.cuda:
        num_gpu = torch.cuda.device_count()
        device = torch.device("cuda:" + str(idx % num_gpu))
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + idx)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + idx)

    env = make_env(args.env, seed=args.seed+idx, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=True)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_net.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_net.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    
    net = AC_LSTM(env.observation_space.shape[0], env.action_space.n).to(device)
    net.train()

    state = env.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(1, 512).to(device)
    cx = torch.zeros(1, 512).to(device)

    while T.value < args.num_timesteps:
        # Reset gradients
        loss_value_v = torch.zeros(1, 1).to(device)
        loss_policy_v = torch.zeros(1, 1).to(device)
        loss_entropy_v = torch.zeros(1, 1).to(device)
        gae_v = torch.zeros(1, 1).to(device)

        hx.detach_()
        cx.detach_()

        rewards = []
        value_vs = []
        log_prob_action_vs = []
        entropy_vs = []

        # Synchronize thread-specific parameters
        net.load_state_dict(shared_net.state_dict())

        for step in range(REWARD_STEPS):
            # Perform action according to policy
            value_v, logit_v, (hx, cx) = net(state_v.unsqueeze(0), (hx, cx))
            prob_v = F.softmax(logit_v, dim=1)
            action_v = prob_v.multinomial(num_samples=1)
            action = int(action_v.item())

            log_prob_v = F.log_softmax(logit_v, dim=1)
            log_prob_action_v = log_prob_v.gather(1, action_v)
            entropy_v = -(log_prob_v * prob_v).sum(dim=1)

            # Receive reward and new state
            state, reward, done, info = env.step(action)
            state_v = torch.from_numpy(state).float().to(device)

            rewards.append(reward)
            value_vs.append(value_v)
            log_prob_action_vs.append(log_prob_action_v)
            entropy_vs.append(entropy_v)

            with lock:
                T.value += 1
            if done:
                state = env.reset()
                state_v = torch.from_numpy(state).float().to(device)
                hx = torch.zeros(1, 512).to(device)
                cx = torch.zeros(1, 512).to(device)
                break

        # R
        R_v = (1 - done) * net(state_v.unsqueeze(0), (hx, cx))[0]
        value_vs.append(R_v)

        for i in reversed(range(len(rewards))):
            R_v = GAMMA * R_v + rewards[i]
            # Accumulate gradients
            adv_v = R_v.detach() - value_vs[i]

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + GAMMA * value_vs[i + 1] - value_vs[i]
            gae_v = gae_v * GAMMA * TAU + delta_t

            loss_value_v += 0.5 * adv_v.pow(2)
            loss_policy_v -= log_prob_action_vs[i] * gae_v.detach()  # cautious: detach()
            loss_entropy_v -= entropy_vs[i]

        net.zero_grad()
        loss_v = COEF_VALUE * loss_value_v + loss_policy_v + COEF_ENTROPY * loss_entropy_v
        loss_v.backward()

        for shared_param, param in zip(shared_net.parameters(), net.parameters()):
            shared_param.grad = param.grad.cpu()

        optimizer.step()
