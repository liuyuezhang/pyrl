from common.experience_replay import Experience, ExperienceReplay

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


########################################
# deep Q-learning with experience replay
########################################
def train(args, env, model, logger, device):
    # Initialize replay memory D to capacity N
    replay_memory = ExperienceReplay(args.buffer_size)

    # Initialize action-value function Q with random weights theta
    net = model(env.observation_space.shape[0], env.action_space.n).to(device)

    # Initialize target action-value function Q hat with weights theta_ = theta
    tgt_net = model(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    logger.add_model(net)

    t = 0
    epsilon = args.epsilon_start

    while t < args.total_steps:
        # Initialize sequence and preprocessed sequence (wrapped in env)
        state = env.reset()
        done = False

        while not done:
            # With probability epsilon select a random action
            if np.random.random() < epsilon or t < args.init_steps:
                action = np.random.choice(env.action_space.n)
            # otherwise select a = argmax Q
            else:
                state_v = torch.from_numpy(state).float().to(device)
                Q_v = net(state_v.unsqueeze(0))
                action_v = torch.argmax(Q_v, dim=1)
                action = int(action_v.item())

            # Execute action in emulator and observe reward and next_state
            next_state, reward, done, info = env.step(action)
            logger.log(t, info, done)
            # logger.log(t, reward, done)
            t += 1

            # Store transition in D
            replay_memory.append(Experience(state, action, reward, done, next_state))

            # Set state = next_state
            state = next_state

            if t >= args.init_steps:
                epsilon = max(args.epsilon_final, args.epsilon_start - (t - args.init_steps) / args.epsilon_steps)
                if t % args.update_freq == 0:
                    for i in range(args.update_steps):
                        # Sample random minibatch of transitions from D
                        states, actions, rewards, dones, next_states = replay_memory.sample(args.batch_size)

                        states_v = torch.from_numpy(states).float().to(device)
                        actions_v = torch.from_numpy(actions).long().to(device)
                        rewards_v = torch.from_numpy(rewards).float().to(device)
                        dones_v = torch.from_numpy(dones).float().to(device)
                        next_states_v = torch.from_numpy(next_states).float().to(device)

                        # Set y
                        Qs_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                        Q_hats_v = tgt_net(next_states_v)
                        ys_v = rewards_v + (1 - dones_v) * args.gamma * torch.max(Q_hats_v, dim=1)[0]
                        ys_v.detach_()

                        # Perform a gradient descent step
                        optimizer.zero_grad()
                        loss = nn.MSELoss()(ys_v, Qs_v)
                        loss.backward()
                        optimizer.step()

                # Every C steps reset Q_hat = Q
                if t % args.target_freq == 0:
                    tgt_net.load_state_dict(net.state_dict())
