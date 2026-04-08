#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 

def normalized_column_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        in_size = np.prod(weight_shape[1:4])
        out_size = np.prod(weight_shape[2:4]) * weight_shape[0]
        # Xavier-style bound: sqrt(6 / (fan_in + fan_out))
        w_bounds = np.sqrt(6.0 / (in_size + out_size))
        m.weight.data.uniform_(-w_bounds, w_bounds)
        m.bias.data.fill_(0)
        
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        in_size = weight_shape[1]
        out_size = weight_shape[0]
        # Xavier-style bound: sqrt(6 / (fan_in + fan_out))
        w_bounds = np.sqrt(6.0 / (in_size + out_size))
        m.weight.data.uniform_(-w_bounds, w_bounds)
        m.bias.data.fill_(0)
        
        
class ActorCritic(nn.Module):
    
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        conv_out_size = self._get_conv_out((num_inputs, 42, 42))
        self.fc = nn.Linear(conv_out_size, 256)

        num_outputs = action_space.n

        # 🔥 Dueling heads
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_outputs)

        # 🔥 Actor head (ajout minimal nécessaire)
        self.actor = nn.Linear(256, num_outputs)

        self.apply(weights_init)

        self.advantage.weight.data = normalized_column_initializer(self.advantage.weight.data, std=0.01)
        self.value.weight.data = normalized_column_initializer(self.value.weight.data, std=1.0)
        self.actor.weight.data = normalized_column_initializer(self.actor.weight.data, std=0.01)

        self.advantage.bias.data.fill_(0)
        self.value.bias.data.fill_(0)
        self.actor.bias.data.fill_(0)

        self.train()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            return int(np.prod(x.size()[1:]))

    def forward(self, inputs):
        x = inputs

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = F.elu(self.fc(x))

        # séparation
        value = self.value(x)
        advantage = self.advantage(x)

        # correction mathématique importante
        advantage = advantage - advantage.mean(dim=1, keepdim=True)

        q = value + advantage

        # 🔥 policy (actor séparé pour A3C)
        policy = self.actor(x)

        return value, policy
