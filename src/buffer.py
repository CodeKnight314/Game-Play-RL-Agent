from collections import deque
import torch
import random 
import numpy as np
from typing import Tuple
import time

class ReplayBuffer:
    def __init__(self, memory: int = 1e6):
        self.memory = deque(maxlen=int(memory))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if isinstance(action, torch.Tensor):
            action_tensor = action.clone().detach().to(self.device).long()
        else:
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
            
        reward_tensor = torch.tensor(reward, dtype=torch.float32).clamp(-1.0, 1.0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)
        
        self.memory.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))
        
    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        states, actions, rewards, next_states, dones = batch
        
        states = torch.stack(states, dim=0).to(self.device)
        actions = torch.stack(actions, dim=0).unsqueeze(-1).to(self.device)
        rewards = torch.stack(rewards, dim=0).unsqueeze(-1).to(self.device)
        next_states = torch.stack(next_states, dim=0).to(self.device)
        dones = torch.stack(dones, dim=0).unsqueeze(-1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
class ReplayBufferFast:
    def __init__(self, state_shape: Tuple[int] = (4, 84, 84), max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, ), dtype=np.long)
        self.rewards = np.zeros((max_size, ), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.dones = np.zeros((max_size, ), dtype=np.bool)
        
    def __len__(self):
        return self.size
    
    def push(self, state: np.array, action: float, reward: float, next_state: np.array, done: bool):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = np.clip(reward, -1.0, 1.0)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.tensor(self.states[indices], dtype=torch.float32).div(255.0).to(self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32).div(255.0).to(self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        return states, actions, rewards, next_states, dones

def benchmark(buffer_class, state_dim=(4, 84, 84), batch_size=256, memory_size=100000, samples=1000):
    buffer = buffer_class(state_dim) if buffer_class == ReplayBufferFast else buffer_class(memory_size)
    
    from tqdm import tqdm
    for _ in tqdm(range(memory_size)):
        state = np.random.randn(*state_dim).astype(np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.randn(*state_dim).astype(np.float32)
        done = np.random.rand() > 0.5
        buffer.push(state, action, reward, next_state, done)

    start = time.time()
    for _ in range(samples):
        buffer.sample(batch_size)
    end = time.time()
    
    return end - start

if __name__ == "__main__":
    print("Deque Time:", benchmark(ReplayBuffer, memory_size=1000000, samples=10000))
    print("Numpy Time:", benchmark(ReplayBufferFast, memory_size=1000000, samples=10000))
