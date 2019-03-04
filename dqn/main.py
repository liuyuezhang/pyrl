from dqn.env import make_env
from dqn.model import DQN

import argparse
import os
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from common.logger import Logger


TRAIN_STEP = 50e6
INIT_STEP = 50000

BATCH_SIZE = 32
BUFFER_SIZE = 1000000

GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_FREQ = 10000
UPDATE_FREQ = 4

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEP = 1e6

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="breakout", help="Name of the environment, default=breakout")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, default=0")
    parser.add_argument("--cuda", default=True, help="Enable cuda, default=True")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    path = './res/' + time.strftime('%Y%m%d') + '-' + args.env + '-' + 'dqn' + '-' + str(args.seed) + '/'
    os.makedirs(path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    env = make_env(env=args.env, seed=args.seed, repeat_act=0.0)

    t = 0
    epsilon = EPSILON_START

    idx = 0
    logger = Logger(name=str(idx), path=path)
    best_mean_reward = float('-inf')

    ########################################
    # deep Q-learning with experience replay
    ########################################
    # Initialize replay memory D to capacity N
    replay_buffer = ExperienceBuffer(BUFFER_SIZE)

    # Initialize action-value function Q with random weights theta
    net = DQN(env.ob_space, len(env.act_set)).to(device)
    print(net)

    # Initialize target action-value function Q hat with weights theta_ = theta
    tgt_net = DQN(env.ob_space, len(env.act_set)).to(device)
    tgt_net.load_state_dict(net.state_dict())

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    while t < int(INIT_STEP + TRAIN_STEP):
        # Initialize sequence and preprocessed sequence (wrapped in env)
        env.reset()
        done = env.done()
        while not done:
            state = env.ob()

            # With probability epsilon select a random action
            if np.random.random() < epsilon or t < INIT_STEP:
                action = np.random.choice(len(env.act_set))
            # otherwise select a = argmax Q
            else:
                state_a = np.array([state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                Q_v = net(state_v)
                action_v = torch.argmax(Q_v, dim=1)
                action = int(action_v.item())

            # Execute action in emulator and observe reward and next_state
            reward = env.act(env.act_set[action])
            done = env.done()
            next_state = env.ob()

            eps_reward = logger.log(t, reward, done)
            if done:
                print(t, eps_reward)
                mean_reward = logger.mean_reward(span=100)
                if mean_reward > best_mean_reward:
                    torch.save(net.state_dict(), path + str(idx) + ".dat")
                    print("Best mean reward updated %.3f -> %.3f, model %d saved" %
                          (best_mean_reward, mean_reward, idx))
                    best_mean_reward = mean_reward

            t += 1

            # Store transition in D
            experience = Experience(state, action, reward, done, next_state)
            replay_buffer.append(experience)

            if t >= INIT_STEP:
                epsilon = max(EPSILON_FINAL, EPSILON_START - (t - INIT_STEP) / EPSILON_DECAY_STEP)

                if t % UPDATE_FREQ == 0:
                    # Sample random minibatch of transitions from D
                    states, actions, rewards, dones, next_states = replay_buffer.sample(BATCH_SIZE)

                    states_v = torch.tensor(states).to(device)
                    actions_v = torch.tensor(actions).to(device)
                    rewards_v = torch.tensor(rewards).to(device)
                    dones_v = torch.tensor(dones).to(device)
                    next_states_v = torch.tensor(next_states).to(device)

                    # Set y
                    Qs_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                    Q_hats_v = tgt_net(next_states_v)
                    ys_v = rewards_v + (1 - dones_v).float() * GAMMA * torch.max(Q_hats_v, dim=1)[0]
                    ys_v.detach_()

                    # Perform a gradient descent step
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(ys_v, Qs_v)
                    loss.backward()
                    optimizer.step()

                # Every C steps reset Q_hat = Q
                if t % TARGET_FREQ == 0:
                    tgt_net.load_state_dict(net.state_dict())


# import cv2
#
# while t < 200000:
#     envs.reset()
#     state = envs.ob()
#     done = False
#
#     eps_reward = 0.0
#     while not done:
#         a = envs.act_set[np.random.choice(len(envs.act_set))]
#         reward = envs.act(a)
#         next_state = envs.ob()
#         done = envs.done()
#         t += 1
#
#         screen = state[0]
#         cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
#         cv2.imshow("screen", screen)
#         if cv2.waitKey(50) & 0xFF == ord('q'):
#             break
#
#         eps_reward += reward
#         state = next_state
#     print("Episode " + " ended with score: " + str(eps_reward))
