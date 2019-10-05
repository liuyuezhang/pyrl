import logging
import time
import numpy as np
import torch


formatter = logging.Formatter('%(message)s')


class VecLogger:
    def __init__(self, path, n, name='test', span=100, print_log=True, print_freq=4, save_model=True):
        """Function setup as many loggers as you want"""
        handler = logging.FileHandler(path + name + '.txt')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.logger = logger
        self.logger.info("t,r,info")

        self.n = n
        self.name = name
        self.path = path
        self.models = []
        self.start_time = time.time()
        self.span = span
        self.print_log = print_log
        self.print_freq = print_freq
        self.save_model = save_model

        self.eps_reward = np.zeros((self.n, 1))
        self.eps_length = np.zeros((self.n, 1))
        self.eps_cnt = 0
        self.reward_list = []
        self.best_mean_reward = float('-inf')

    def add_model(self, model):
        self.models.append(model)

    def log(self, t, reward, done, info=None):
        self.eps_reward += reward
        self.eps_length += np.ones((self.n, 1))

        if np.sum(done) > 0:
            avr_eps_reward = np.mean(self.eps_reward[done])
            avr_eps_length = np.mean(self.eps_length[done])

            self.logger.info("{},{},{}".format(t, avr_eps_reward, info))
            self.reward_list.append(avr_eps_reward)
            self.eps_cnt += 1

            if self.print_log and self.eps_cnt % self.print_freq == 0:
                print("Time {}, num steps {}, FPS {:.0f}, reward {}, length {}, {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)), t,
                    t / (time.time() - self.start_time), avr_eps_reward, avr_eps_length, info))

            if self.save_model and self.eps_cnt > self.span:
                mean_reward = np.mean(self.reward_list[-self.span:])
                if mean_reward > self.best_mean_reward:
                    for model in self.models:
                        torch.save(model.state_dict(), self.path + model.__class__.__name__ + ".dat")
                    print("Model: best mean reward updated {:.3f} -> {:.3f}, model saved".format(
                        self.best_mean_reward, mean_reward))
                    self.best_mean_reward = mean_reward

            # reset
            self.eps_reward[done] = 0
            self.eps_length[done] = 0
