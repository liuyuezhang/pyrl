from envs.atari.env import make_env

from common.vec_env.subproc_vec_env import SubprocVecEnv

import torch
import torch.nn.functional as F


GAMMA = 0.99
TAU = 1.00
REWARD_STEPS = 20
CLIP_GRAD = 50

COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(args, T, lock, net, optimizer):
    n = args.num_processes

    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Environment vectorization, automatic reset() in SubprocVecEnv
    env_fns = []
    for idx in range(1, n + 1):
        env_fns.append(lambda: make_env(args.env, seed=args.seed + idx))
    venv = SubprocVecEnv(env_fns)

    state = venv.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(n, 512).to(device)
    cx = torch.zeros(n, 512).to(device)

    while T.value < args.num_timesteps:
        # Reset gradients
        loss_value_v = torch.zeros(1, 1).to(device)
        loss_policy_v = torch.zeros(1, 1).to(device)
        loss_entropy_v = torch.zeros(1, 1).to(device)
        gae_v = torch.zeros(n, 1).to(device)

        hx.detach_()
        cx.detach_()

        reward_vs = []
        done_vs = []
        value_vs = []
        log_prob_action_vs = []
        entropy_vs = []

        for step in range(REWARD_STEPS):
            # Perform action according to policy
            value_v, logit_v, (hx, cx) = net(state_v, (hx, cx))
            prob_v = F.softmax(logit_v, dim=1)
            action_v = prob_v.multinomial(num_samples=1)
            action = action_v.numpy()

            log_prob_v = F.log_softmax(logit_v, dim=1)
            log_prob_action_v = log_prob_v.gather(1, action_v)
            entropy_v = -(log_prob_v * prob_v).sum(dim=1, keepdim=True)

            # Receive reward and new state
            state, reward, done, info = venv.step(action)
            state_v = torch.from_numpy(state).float().to(device)
            reward_v = torch.from_numpy(reward).float().unsqueeze(1).to(device)
            done_v = torch.from_numpy(done.astype('int')).float().unsqueeze(1).to(device)

            reward_vs.append(reward_v)
            done_vs.append(done_v)
            value_vs.append(value_v)
            log_prob_action_vs.append(log_prob_action_v)
            entropy_vs.append(entropy_v)

            with lock:
                T.value += n

        # R
        R_v = (1 - done_v) * net(state_v, (hx, cx))[0]
        value_vs.append(R_v)

        for i in reversed(range(len(reward_vs))):
            R_v = (1 - done_vs[i]) * GAMMA * R_v + reward_vs[i]
            # Accumulate gradients
            adv_v = R_v.detach() - value_vs[i]

            # Generalized Advantage Estimataion
            delta_t = reward_vs[i] + (1 - done_vs[i]) * GAMMA * value_vs[i + 1] - value_vs[i]
            gae_v = gae_v * (1 - done_vs[i]) * GAMMA * TAU + delta_t

            loss_value_v += (0.5 * adv_v.pow(2)).sum()
            loss_policy_v -= (log_prob_action_vs[i] * gae_v.detach()).sum()  # cautious: detach()
            loss_entropy_v -= (entropy_vs[i]).sum()

        net.zero_grad()
        loss_v = COEF_VALUE * loss_value_v + loss_policy_v + COEF_ENTROPY * loss_entropy_v
        loss_v.backward()

        optimizer.step()

    venv.close()
