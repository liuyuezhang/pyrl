from a3c.env import make_env
from a3c.model import A3C

import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
REWARD_STEPS = 5
# CLIP_GRAD = 50

COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(idx, args, T, shared_net, optimizer):
    if args.cuda:
        num_gpu = torch.cuda.device_count()
        device = torch.device("cuda:" + str(idx % num_gpu))
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + idx)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + idx)

    env = make_env(args.env, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=True)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_net.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_net.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + idx)
    
    net = A3C(env.observation_space.shape[0], env.action_space).to(device)
    net.train()

    state = env.reset()
    state_v = torch.from_numpy(state).float().to(device)

    while T.value < args.num_timesteps:
        # Reset gradients
        loss_value_v = torch.zeros(1, 1).to(device)
        loss_policy_v = torch.zeros(1, 1).to(device)
        loss_entropy_v = torch.zeros(1, 1).to(device)

        rewards = []
        value_vs = []
        log_prob_action_vs = []
        entropy_vs = []

        # Synchronize thread-specific parameters
        net.load_state_dict(shared_net.state_dict())

        for step in range(REWARD_STEPS):
            # Perform action according to policy
            value_v, logit_v = net(state_v.unsqueeze(0))
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

            T.value += 1
            if done:
                state = env.reset()
                state_v = torch.from_numpy(state).float().to(device)
                break

        # R
        R_v = torch.zeros(1, 1).to(device) if done else net(state_v.unsqueeze(0))[0]
        value_vs.append(R_v)

        for i in reversed(range(len(rewards))):
            R_v = GAMMA * R_v + rewards[i]
            # Accumulate gradients
            adv_v = R_v.detach() - value_vs[i]
            loss_value_v += 0.5 * adv_v.pow(2)
            loss_policy_v -= log_prob_action_vs[i] * adv_v.detach()  # cautious: detach()
            loss_entropy_v -= entropy_vs[i]

        net.zero_grad()
        loss_v = COEF_VALUE * loss_value_v + loss_policy_v + COEF_ENTROPY * loss_entropy_v
        loss_v.backward()

        for shared_param, param in zip(shared_net.parameters(), net.parameters()):
            shared_param.grad = param.grad.cpu()

        optimizer.step()
