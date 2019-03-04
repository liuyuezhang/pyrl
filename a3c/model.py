import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.linear1 = nn.Linear(3136, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_space.n)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)
