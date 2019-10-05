from common.vec_env.vec_logger import VecLogger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
TAU = 1.00
REWARD_STEPS = 20
CLIP_GRAD = 50

COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(args, venv, model, path, device):
    n = args.num_processes

    net = model(venv.observation_space.shape[0], venv.action_space.n).to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    vlogger = VecLogger(n=n, path=path)
    vlogger.add_model(net)

    state = venv.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(n, 512).to(device)
    cx = torch.zeros(n, 512).to(device)

    t = 0

    while t < args.num_timesteps:
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
            action = action_v.cpu().numpy()

            log_prob_v = F.log_softmax(logit_v, dim=1)
            log_prob_action_v = log_prob_v.gather(1, action_v)
            entropy_v = -(log_prob_v * prob_v).sum(dim=1, keepdim=True)

            # Receive reward and new state
            state, reward, done, info = venv.step(action)
            t += n

            reward = np.expand_dims(reward, axis=1)
            done = np.expand_dims(done, axis=1)
            info = np.expand_dims(info, axis=1)
            vlogger.log(t, reward, info)

            state_v = torch.from_numpy(state).float().to(device)
            reward_v = torch.from_numpy(reward).float().to(device)
            done_v = torch.from_numpy(done.astype('int')).float().to(device)

            # Reset the LSTM state if done
            hx = (1 - done_v) * hx
            cx = (1 - done_v) * cx

            reward_vs.append(reward_v)
            done_vs.append(done_v)
            value_vs.append(value_v)
            log_prob_action_vs.append(log_prob_action_v)
            entropy_vs.append(entropy_v)

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
        nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

        optimizer.step()

    venv.close()
