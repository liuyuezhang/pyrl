# PyRL
PyRL (pronounced "Parallel") is a PyTorch deep reinforcement learning 
library focusing on **reproducibility** and **readability**.

Our philosophy is to respect all the details in 
original papers. We wish to keep it highly-readable by 
following the pseudocode and implementing with PyTorch.

Let's make deep RL easier to start!

## Subpackages
Currently, PyRL includes implementations of:
* [DQN](dqn) ([Nature 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf))
* [A3C](a3c) ([ICML 2016](https://arxiv.org/pdf/1602.01783.pdf))
* [DDPG](ddpg) ([ICML 2016](https://arxiv.org/abs/1509.02971), on the list)
* [PPO](ppo) ([arXiv 2017](https://arxiv.org/abs/1707.06347), ongoing)
* [RAINBOW](rainbow) ([AAAI 2018](https://arxiv.org/pdf/1710.02298.pdf), on the list)

More methods may be included in the future. 

## Prerequisites
* Python 3.6
* PyTorch 1.0.1
* OpenAI Gym 0.10.9

## Usage
To train an agent using A3C, come to the root directory, 
and simply run:
```sh
python -m a3c.main
```
which train an agent to solve the game Breakout with random seed 0. 

Results will be saved in [res/BreakoutNoFrameskip-v4_a3c_0/](res/BreakoutNoFrameskip-v4_a3c_0) by default.

To evaluate the trained A3C model, run:
```sh
python -m a3c.eval
```

More options could be found in arguments (see [main.py](a3c/main.py) and [eval.py](a3c/eval.py)).

## Structure
The PyRL package contains: 
* method packages, e.g. [a3c/](a3c)
* environment packages, [envs/](envs)
* common package, [common/](common)
* result directory, [res/](res)
* plot script, [plot.ipynb](plot.ipynb)

Each method packages and environment packages are designed to work independently. 
Therefore, if you would like to use A3C on atari games, you will only need [a3c/](a3c) 
and [envs/atari/](envs/atari), as well as [common/](common), [res/](res), [plot.ipynb](plot.ipynb).

### Method Packages
Each method package (e.g. [a3c/](a3c)) contains:
* [main.py](a3c/main.py), main file for training
* [model.py](a3c/model.py), model file defines the network architecture
* [eval.py](a3c/eval.py), evaluation file evaluates the performance of the saved model
* other files


### Environment Packages
Environment packages ([envs/](envs)) contain:
* [atari/](envs/atari), atari games environment based on OpenAI Gym

### Common Package
Common package ([common/](common)) contains:
* [logger.py](common/logger.py), logger to log training and evaluation data

### Result Directory
The result directory ([res/](res)) follows the naming rule: 

```
[env_id]_[method]_[seed]
```

e.g. [BreakoutNoSkipFrame-v4_a3c_0](res/BreakoutNoSkipFrame-v4_a3c_0), contains:

* [test.txt](), logging file during training, generated by [main.py](a3c/main.py)
* [model.dat](), best saved model during training, generated by [main.py](a3c/main.py)
* [eval.txt](). logging file during evaluation, generated by [eval.py](a3c/eval.py)


### Plot Script
* [plot.ipynb](plot.ipynb), plot learning curves in the result directory, averaged 
across different random seeds

For more details of a specific method, please refer to 
the README in each method packages.

### Citing the Project
To cite this repository in publications:

    @misc{PyRL,
      author = {Yuezhang, Liu},
      title = {PyRL},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/liuyuezhangadam/pyrl}},
    }

### Acknowledgements
I would like to express my great thankness to authors of 
several high quality deep RL implementations. References are provided 
in each independent READMEs. Frankly speaking, I am not familiar with 
LICENSE issues, so please contact me if I made any mistakes on 
copyright affairs.
