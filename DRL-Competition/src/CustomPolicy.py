import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = self.softmax(torch.matmul(queries, keys.transpose(-2, -1)))
        out = torch.matmul(attention_scores, values)
        return out

class CustomPolicyWithAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomPolicyWithAttention, self).__init__(observation_space, features_dim)
        self.mlp = CustomMLP(observation_space.shape[0], features_dim)
        self.attention = SelfAttention(features_dim)

    def forward(self, observations):
        x = self.mlp(observations)
        return self.attention(x)



