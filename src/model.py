import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from torch.distributions import Categorical

class ActorCritc(nn.Module):
    def __init__(self, input_shape: Tuple[int], hidden_dim: int, num_actions: int):
        super().__init__()
        
        self.conv = nn.Sequential(*[
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ])
        
        self.conv_output = self._get_conv_output(input_shape)
        
        self.actor = nn.Sequential(*[
            nn.Linear(self.conv_output, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        ])
        
        self.critic = nn.Sequential(*[
            nn.Linear(self.conv_output, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])
        
        self.apply(self._init_weights)
        
    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        
        action_probs = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        return action_probs, value
    
    def get_action(self, obs, deterministic=False):
        action_logits, value = self(obs)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
            
        action_log_probs = dist.log_prob(action)
        
        return action, action_log_probs, value
    
    def evaluate_actions(self, obs, actions):
        action_logits, value = self(obs)
        dist = Categorical(logits=action_logits)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, value, entropy

class DQN(torch.nn.Module):
    def __init__(self, input_shape, hidden_dim, num_actions):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy_input)
            n_flatten = conv_out.shape[1]
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    def get_action(self, state, epsilon=0.0):
        with torch.no_grad():
            if torch.rand(1) < epsilon:
                return torch.randint(0, self.fc[-1].out_features, (state.shape[0],))
            
            q_values = self(state)
            return q_values.argmax(dim=1)