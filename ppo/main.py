import argparse
import os

import torch
from common.logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO-LSTM')
    parser.add_argument('--env',               type=str,   default="LunarLander-v2")
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--num-steps',         type=int,   default=15e6)
    parser.add_argument('--max-episode-steps', type=int,   default=300)
    parser.add_argument('--num-processes',     type=int,   default=16)

    parser.add_argument('--lr',                type=float, default=2e-3)
    parser.add_argument('--gamma',             type=float, default=0.99)
    parser.add_argument('--T',                 type=int,   default=2000)
    parser.add_argument('--K_epochs',           type=int,   default=4)
    parser.add_argument('--eps_clip',          type=float, default=0.2)

    parser.add_argument('--cuda',                          default=True)
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

    # path
    path = './res/' + args.env + '_' + 'ppo' + '_' + str(args.seed) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # device
    device = torch.device("cuda" if args.cuda else "cpu")

    # env
    import gym
    env = gym.make(args.env)
    env._max_episode_steps = args.max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # agent
    from ppo.agent import PPO
    agent = PPO(state_dim, action_dim, lr=args.lr, gamma=args.gamma, K_epochs=args.K_epochs,
                eps_clip=args.eps_clip, device=device)
    
    # logger
    logger = Logger(path=path, print_freq=100)
    logger.add_model(agent.policy)
    
    # training loop
    t = 0
    while t < args.num_steps:
        state = env.reset()
        for _ in range(args.T):
            # Running policy_old
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            t += 1
            logger.log(t, reward, done)
            
            # Saving reward and done
            agent.step(reward, done)

            if done:
                state = env.reset()

        # update if its time
        agent.update()
