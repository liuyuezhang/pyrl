from common.vec_env.vec_logger import VecLogger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# batch_size = trajectory_steps * num_processes
# num_processes_minibatch = num_processes // num_minibatch
GAMMA = 0.99
TAU = 0.95
TRAJECTORY_STEPS = 5
NUM_EPOCH = 2
NUM_MINIBATCH = 1
CLIP_GRAD = 50

PPO_CLIP = 0.2
COEF_VALUE = 0.5
COEF_ENTROPY = 0.01


def train(args, venv, model, path, device):
    N = args.num_processes
    T = TRAJECTORY_STEPS
    K = NUM_EPOCH
    M = NUM_MINIBATCH
    assert N % M == 0

    net = model(venv.observation_space.shape[0], venv.action_space.n).to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    vlogger = VecLogger(n=N, path=path)
    vlogger.add_model(net)

    state = venv.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(N, 512).to(device)
    cx = torch.zeros(N, 512).to(device)

    t = 0

    while t < args.num_timesteps:
        # Reset gradients
        loss_value_v = torch.zeros(1, 1).to(device)
        loss_policy_v = torch.zeros(1, 1).to(device)
        gae_v = torch.zeros(M, 1).to(device)

        # seperate hx, cx and hx_v, cx_v
        hx.detach_()
        cx.detach_()

        # change to numpy while running
        m_states =
        m_rewards =
        m_dones =
        m_hxs =
        m_cxs =
        m_old_log_prob_actions =

        # save state, hx, cx, before perform
        m_state_vs = torch.cat((m_state_vs, state_v.unsqueeze(0)))
        m_hx_s = torch.cat((m_hx_s, hx.unsqueeze(0)))
        m_cx_s = torch.cat((m_cx_s, cx.unsqueeze(0)))

        for step in range(TRAJECTORY_STEPS):
            with torch.no_grad():
                # Perform action according to policy
                value_v, logit_v, (hx, cx) = net(state_v, (hx, cx))
                prob_v = F.softmax(logit_v, dim=1)
                action_v = prob_v.multinomial(num_samples=1)
                action = action_v.data.cpu().numpy()

                log_prob_v = F.log_softmax(logit_v, dim=1)
                log_prob_action_v = log_prob_v.gather(1, action_v)

                # Receive reward and new state
                state, reward, done, info = venv.step(action)
                t += N

                reward = np.expand_dims(reward, axis=1)
                done = np.expand_dims(done, axis=1)
                info = np.expand_dims(info, axis=1)
                vlogger.log(t, reward, info)

                state_v = torch.from_numpy(state).float().to(device)
                reward_v = torch.from_numpy(reward).float().to(device)
                done_v = torch.from_numpy(done.astype('int')).float().to(device)

                m_state_vs = torch.cat((m_state_vs, state_v.unsqueeze(0)))
                m_reward_vs = torch.cat((m_reward_vs, reward_v.unsqueeze(0)))
                m_done_vs = torch.cat((m_done_vs, done_v.unsqueeze(0)))
                m_hx_s = torch.cat((m_hx_s, hx.unsqueeze(0)))
                m_cx_s = torch.cat((m_cx_s, cx.unsqueeze(0)))
                m_old_log_prob_action_vs = torch.cat((m_old_log_prob_action_vs, log_prob_action_v.unsqueeze(0)))

                # Reset the LSTM state if done
                hx = (1 - done_v) * hx
                cx = (1 - done_v) * cx

        for _ in range(K):
            perm = torch.randperm(N).to(device)
            for k in range(0, N, M):
                # select a minibatch of processes to get a minibatch of state_vs
                indices = perm[k:k + M]

                # minibatch tensors shape (T, num_processes_minibatch, ...)
                state_vs = torch.index_select(m_state_vs, 1, indices)
                reward_vs = torch.index_select(m_reward_vs, 1, indices)
                done_vs = torch.index_select(m_done_vs, 1, indices)
                hx_s = torch.index_select(m_hx_s, 1, indices)
                cx_s = torch.index_select(m_cx_s, 1, indices)
                old_log_prob_action_vs = torch.index_select(m_old_log_prob_action_vs, 1, indices)

                value_vs = torch.Tensor([]).to(device)
                logit_vs = torch.Tensor([]).to(device)
                for j in range(0, M):
                    value_v, logit_v, _ = net(state_vs[:, j, ...], (hx_s[:, j, ...], cx_s[:, j, ...]))
                    value_vs = torch.cat((value_vs, value_v.unsqueeze(1)), dim=1)
                    logit_vs = torch.cat((logit_vs, logit_v.unsqueeze(1)), dim=1)

                logit_vs = logit_vs[:-1, ...].view(T * M, -1)
                prob_vs = F.softmax(logit_vs, dim=1)
                action_vs = prob_vs.multinomial(num_samples=1)
                log_prob_vs = F.log_softmax(logit_vs, dim=1)
                log_prob_action_vs = log_prob_vs.gather(1, action_vs)
                log_prob_action_vs = log_prob_action_vs.view(T, M, -1)

                # R
                R_v = (1 - done_vs[-1]) * value_vs[-1]
                for i in reversed(range(T)):
                    R_v = (1 - done_vs[i]) * GAMMA * R_v + reward_vs[i]
                    # Accumulate gradients
                    adv_v = R_v.detach() - value_vs[i]

                    # # Generalized Advantage Estimataion
                    # delta_t = reward_vs[i] + (1 - done_vs[i]) * GAMMA * value_vs[i + 1] - value_vs[i]
                    # gae_v = gae_v * (1 - done_vs[i]) * GAMMA * TAU + delta_t
                    #
                    # ratio = torch.exp(log_prob_action_vs[i] - old_log_prob_action_vs[i])
                    # surr1 = ratio * gae_v.detach()
                    # surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * gae_v.detach()

                    loss_value_v += (0.5 * adv_v.pow(2)).sum()
                    # loss_policy_v -= torch.min(surr1, surr2).sum()  # cautious: detach()

                loss_entropy_v = (log_prob_vs * prob_vs).sum()

                net.zero_grad()
                loss_v = COEF_VALUE * loss_value_v # + loss_policy_v + COEF_ENTROPY * loss_entropy_v
                loss_v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

                optimizer.step()

    venv.close()
