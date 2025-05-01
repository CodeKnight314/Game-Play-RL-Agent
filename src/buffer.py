from collections import deque
import torch
import random 
import numpy as np

class ReplayBuffer:
    def __init__(self, memory: int = 1e6):
        self.memory = deque(maxlen=int(memory))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.float32)
        
        self.memory.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))
        
    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        states, actions, rewards, next_states, dones = batch
        
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0).unsqueeze(-1)
        rewards = torch.stack(rewards, dim=0).unsqueeze(-1)
        next_states = torch.stack(next_states, dim=0)
        dones = torch.stack(dones, dim=0).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones
        
        