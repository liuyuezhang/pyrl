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
    hx_v = torch.zeros((N, 512)).to(device)
    cx_v = torch.zeros((N, 512)).to(device)

    t = 0

    while t < args.num_timesteps:
        # change to numpy while running
        m_states = []
        m_rewards = []
        m_dones = []
        m_hxs = []
        m_cxs = []
        m_old_log_prob_actions = []

        for step in range(T):
            # save state, hx, cx, before perform
            m_states.append(state)
            hx = hx_v.data.cpu().numpy()
            cx = cx_v.data.cpu().numpy()
            m_hxs.append(hx)
            m_cxs.append(cx)

            # Perform action according to policy
            state_v = torch.from_numpy(state).float().to(device)
            value_v, logit_v, (hx_v, cx_v) = net(state_v, (hx_v, cx_v))
            prob_v = F.softmax(logit_v, dim=1)
            action_v = prob_v.multinomial(num_samples=1)
            log_prob_v = F.log_softmax(logit_v, dim=1)
            log_prob_action_v = log_prob_v.gather(1, action_v)

            # Receive reward and new state
            action = action_v.data.cpu().numpy()
            state, reward, done, info = venv.step(action)
            t += N

            reward = np.expand_dims(reward, axis=1)
            done = np.expand_dims(done, axis=1)
            info = np.expand_dims(info, axis=1)
            vlogger.log(t, reward, info)

            m_rewards.append(reward)
            m_dones.append(done)
            log_prob_action = log_prob_action_v.data.cpu().numpy()
            m_old_log_prob_actions.append(log_prob_action)

            # Reset the LSTM state if done
            done_v = torch.from_numpy(done.astype('int')).float().to(device)
            hx_v = (1 - done_v) * hx_v
            cx_v = (1 - done_v) * cx_v

        m_states = np.array(m_states)
        m_hxs = np.array(m_hxs)
        m_cxs = np.array(m_cxs)
        m_rewards = np.array(m_rewards)
        m_dones = np.array(m_dones, dtype='int')
        m_old_log_prob_actions = np.array(m_old_log_prob_actions)

        for _ in range(K):
            indices = np.random.permutation(N)
            for k in range(0, N, M):
                # Reset gradients
                loss_value_v = torch.zeros(1, 1).to(device)
                loss_policy_v = torch.zeros(1, 1).to(device)
                gae_v = torch.zeros(M, 1).to(device)

                # minibatch tensors shape (T, M, ...)
                state_vs = torch.from_numpy(m_states[:, indices[k:k+M], ...]).float().view((T*M, 1, 84, 84)).to(device)
                hx_vs = torch.from_numpy(m_hxs[:, indices[k:k+M], ...]).float().view((T*M, -1)).to(device)
                cx_vs = torch.from_numpy(m_cxs[:, indices[k:k+M], ...]).float().view((T*M, -1)).to(device)
                reward_vs = torch.from_numpy(m_rewards[:, indices[k:k+M], ...]).float().to(device)
                done_vs = torch.from_numpy(m_dones[:, indices[k:k+M], ...]).float().to(device)
                old_log_prob_action_vs = torch.from_numpy(m_old_log_prob_actions[:, indices[k:k+M], ...]).float().to(device)

                # reconstruct under new policy
                value_vs, logit_vs, _ = net(state_vs, (hx_vs, cx_vs))
                prob_vs = F.softmax(logit_vs, dim=1)
                action_vs = prob_vs.multinomial(num_samples=1)
                log_prob_vs = F.log_softmax(logit_vs, dim=1)
                log_prob_action_vs = log_prob_vs.gather(1, action_vs)
                log_prob_action_vs = log_prob_action_vs.view(T, M, -1)

                # R
                R_v = (1 - done_vs[-1]) * value_vs[-1]
                for i in reversed(range(T-1)):
                    R_v = (1 - done_vs[i]) * GAMMA * R_v + reward_vs[i]
                    # Accumulate gradients
                    adv_v = R_v.detach() - value_vs[i]

                    # Generalized Advantage Estimataion
                    delta_t = reward_vs[i] + (1 - done_vs[i]) * GAMMA * value_vs[i + 1] - value_vs[i]
                    gae_v = gae_v * (1 - done_vs[i]) * GAMMA * TAU + delta_t

                    ratio = torch.exp(log_prob_action_vs[i] - old_log_prob_action_vs[i])
                    surr1 = ratio * gae_v.detach()
                    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * gae_v.detach()

                    loss_value_v += (0.5 * adv_v.pow(2)).sum()
                    loss_policy_v -= torch.min(surr1, surr2).sum()  # cautious: detach()

                loss_entropy_v = (log_prob_vs * prob_vs).sum()

                net.zero_grad()
                loss_v = COEF_VALUE * loss_value_v + loss_policy_v + COEF_ENTROPY * loss_entropy_v
                loss_v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

                optimizer.step()

    venv.close()
