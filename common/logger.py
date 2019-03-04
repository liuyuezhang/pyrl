import logging
import time
import numpy as np

import torch

formatter = logging.Formatter('%(message)s')


class Logger:
    def __init__(self, name, path, model, start_time, span=100, print_log=True, save_model=True):
        """Function setup as many loggers as you want"""
        handler = logging.FileHandler(path + name + '.txt')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.logger = logger
        self.logger.info("t,r")

        self.name = name
        self.path = path
        self.model = model
        self.start_time = start_time
        self.span = span
        self.print_log = print_log
        self.save_model = save_model

        self.eps_reward = 0.0
        self.eps_length = 0
        self.reward_list = []
        self.best_mean_reward = float('-inf')

    def log(self, t, reward, done):
        self.eps_reward += reward
        self.eps_length += 1
        if done:
            self.logger.info("{},{}".format(t, self.eps_reward))
            self.reward_list.append(self.eps_reward)

            if self.print_log:
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)), t,
                    t / (time.time() - self.start_time), self.eps_reward, self.eps_length))

            if self.save_model:
                mean_reward = np.mean(self.reward_list[-self.span:])
                if mean_reward > self.best_mean_reward:
                    torch.save(self.model.state_dict(), self.path + "model.dat")
                    print("Model: best mean reward updated %.3f -> %.3f, model saved" %
                          (self.best_mean_reward, mean_reward))
                    self.best_mean_reward = mean_reward

            # reset
            self.eps_reward = 0.0
            self.eps_length = 0
