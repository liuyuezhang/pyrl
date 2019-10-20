from common.vec_env.vec_logger import VecLogger

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


GAMMA = 0.99
TAU = 0.95
N_STEPS = 5
NUM_EPOCH = 4
NUM_MINIBATCH = 4
CLIP_GRAD = 0.5

PPO_EPS = 0.2
COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(args, venv, model, path, device):
    N = args.num_processes
    T = N_STEPS
    K = NUM_EPOCH
    M = NUM_MINIBATCH
    assert N % M == 0

    net = model(venv.observation_space.shape[0], venv.action_space.n).to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    vlogger = VecLogger(N=N, path=path)
    vlogger.add_model(net)

    state = venv.reset()
    hx_v = torch.zeros((N, 512)).to(device)
    cx_v = torch.zeros((N, 512)).to(device)

    t = 0

    while t < args.num_timesteps:
        # Run policy pi_old in environment for T timesteps
        with torch.no_grad():
            # use numpy while running, ppo is an off-policy method as acer
            states = []
            hxs = []
            cxs = []
            actions = []
            rewards = []
            dones = []
            values = []
            Rs = []
            gaes = []
            log_prob_actions = []

            # no gradient, otherwise GPU memory will leak
            for step in range(T):
                # save state, hx, cx before perform
                states.append(state.copy())
                hx = hx_v.data.cpu().numpy()
                cx = cx_v.data.cpu().numpy()
                hxs.append(hx.copy())
                cxs.append(cx.copy())

                # perform action according to policy
                state_v = torch.from_numpy(state).float().to(device)
                value_v, logit_v, (hx_v, cx_v) = net(state_v, (hx_v, cx_v))
                dist_v = Categorical(logits=logit_v)
                action_v = dist_v.sample()
                log_prob_action_v = dist_v.log_prob(action_v)
                action_v = action_v.unsqueeze(1)
                log_prob_action_v = log_prob_action_v.unsqueeze(1)

                # receive reward and new state
                action = action_v.data.cpu().numpy()
                state, reward, done, info = venv.step(action)
                t += N

                reward = np.expand_dims(reward, axis=1)
                done = np.expand_dims(done, axis=1)
                info = np.expand_dims(info, axis=1)
                vlogger.log(t, reward, info)

                value = value_v.data.cpu().numpy()
                log_prob_action = log_prob_action_v.data.cpu().numpy()

                actions.append(action.copy())
                rewards.append(reward.copy())
                dones.append(done.copy())
                values.append(value.copy())
                log_prob_actions.append(log_prob_action.copy())

                # reset the LSTM state if done
                done_v = torch.from_numpy(done.astype('int')).float().to(device)
                hx_v = (1 - done_v) * hx_v
                cx_v = (1 - done_v) * cx_v

            # last value
            state_v = torch.from_numpy(state).float().to(device)
            value_v = net(state_v, (hx_v, cx_v))[0]
            value = value_v.data.cpu().numpy()
            values.append(value.copy())

        # Compute advantage estimates
        R = (1 - done) * value
        gae = np.zeros((N, 1))
        for i in reversed(range(T)):
            # reference values
            R = (1 - dones[i]) * GAMMA * R + rewards[i]
            Rs.insert(0, R.copy())
            # generalized advantage estimataion
            delta_t = rewards[i] + (1 - dones[i]) * GAMMA * values[i + 1] - values[i]
            gae = gae * (1 - dones[i]) * GAMMA * TAU + delta_t
            gaes.insert(0, gae.copy())

        states = np.array(states)
        hxs = np.array(hxs)
        cxs = np.array(cxs)
        actions = np.array(actions)
        Rs = np.array(Rs)
        gaes = np.array(gaes)
        log_prob_actions = np.array(log_prob_actions)

        # gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        # Optimize surrogate L wrt theta, with K epochs and minibatch size NM <= NT
        for _ in range(K):
            indices = np.random.permutation(N)
            for k in range(0, N, M):
                # minibatch tensors shape (T*M, ...), main bottleneck for GPU memory
                state_vs = torch.from_numpy(states[:, indices[k:k+M], ...].reshape(T*M, 1, 84, 84)).float().to(device)
                hx_vs = torch.from_numpy(hxs[:, indices[k:k+M], ...].reshape(T*M, -1)).float().to(device)
                cx_vs = torch.from_numpy(cxs[:, indices[k:k+M], ...].reshape(T*M, -1)).float().to(device)
                action_vs = torch.from_numpy(actions[:, indices[k:k+M], ...].reshape(T*M, -1)).float().to(device)
                R_vs = torch.from_numpy(Rs[:, indices[k:k+M], ...].reshape(T*M, -1)).float().to(device)
                gae_vs = torch.from_numpy(gaes[:, indices[k:k + M], ...].reshape(T*M, -1)).float().to(device)
                old_log_prob_action_vs = torch.from_numpy(log_prob_actions[:, indices[k:k+M], ...].reshape(T*M, -1)).float().to(device)

                # reconstruct under new policy using old states and old actions
                value_vs, logit_vs, _ = net(state_vs, (hx_vs, cx_vs))
                dist_vs = Categorical(logits=logit_vs)
                log_prob_action_vs = dist_vs.log_prob(action_vs)
                log_prob_action_vs = log_prob_action_vs.unsqueeze(1)

                adv_v = value_vs - R_vs
                ratio_v = torch.exp(log_prob_action_vs - old_log_prob_action_vs)
                surr_v = ratio_v * gae_vs
                clipped_surr_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS) * gae_vs

                loss_value_v = (0.5 * adv_v.pow(2)).mean()
                loss_policy_v = -torch.min(surr_v, clipped_surr_v).mean()
                loss_entropy_v = dist_vs.entropy().mean()

                net.zero_grad()
                loss_v = COEF_VALUE * loss_value_v + loss_policy_v + COEF_ENTROPY * loss_entropy_v
                loss_v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

                optimizer.step()

    venv.close()
