import torch
import torch.nn as nn
import torch.nn.functional as F


class AC_LSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, debug=False):
        super(AC_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.lstm = nn.LSTMCell(3136, 512)
        self.linear_critic = nn.Linear(512, 1)
        self.linear_actor = nn.Linear(512, num_actions)

        self.debug = debug

    def forward(self, ob, hist):
        hx, cx = hist
        conv1 = F.relu(self.conv1(ob))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        x = conv3.view(conv3.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        if not self.debug:
            return self.linear_critic(hx), self.linear_actor(hx), (hx, cx)
        else:
            return self.linear_critic(hx), self.linear_actor(hx), (hx, cx), conv3.cpu().numpy()
