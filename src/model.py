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
        
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features).squeeze(-1)
        
        return action_probs, value
    
    def get_action(self, obs, deterministic=True):
        action_probs, value = self(obs)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.rsample()
            
        action_log_probs = dist.log_prob(action)
        
        return action, action_log_probs, value
    
    def evaluate_actions(self, obs, actions):
        action_probs, value = self(obs)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, value, entropy