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
        self.actions = np.zeros((max_size, ), dtype=np.longlong)
        self.rewards = np.zeros((max_size, ), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.dones = np.zeros((max_size, ), dtype=np.bool_)
        
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
    
class RolloutBuffer:
    def __init__(self, rollout: int, num_envs: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...] = None, device: str = None):
        self.rollout_length = rollout
        self.num_envs = num_envs
        self.obs_shape = obs_shape

        if action_shape is None:
            action_shape = ()
        elif isinstance(action_shape, int):
            action_shape = (action_shape,)
        self.action_shape = action_shape

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        default_device = 'cpu'
        self.states = torch.zeros((rollout, num_envs, *obs_shape), device=default_device)
        self.actions = torch.zeros((rollout, num_envs, *action_shape), device=default_device)
        self.rewards = torch.zeros((rollout, num_envs), device=default_device)
        self.dones   = torch.zeros((rollout, num_envs), device=default_device, dtype=torch.bool)
        self.next_states = torch.zeros((rollout, num_envs, *obs_shape), device=default_device)
        self.logp_old = torch.zeros((rollout, num_envs), device=default_device)
        self.value_old = torch.zeros((rollout, num_envs), device=default_device)
        self.advantages = torch.zeros_like(self.rewards)
        self.returns    = torch.zeros_like(self.rewards)
        self.t = 0

    def to(self, device: str):
        self.device = device
        for attr in ('states', 'actions', 'rewards', 'dones', 'next_states', 'logp_old', 'value_old', 'advantages', 'returns'):
            setattr(self, attr, getattr(self, attr).to(device))

    def reset(self):
        self.t = 0
        default_device = 'cpu'
        self.states     = torch.zeros((self.rollout_length, self.num_envs, *self.obs_shape), device=default_device)
        self.actions    = torch.zeros((self.rollout_length, self.num_envs, *self.action_shape), device=default_device)
        self.rewards    = torch.zeros((self.rollout_length, self.num_envs), device=default_device)
        self.dones      = torch.zeros((self.rollout_length, self.num_envs), device=default_device, dtype=torch.bool)
        self.next_states = torch.zeros((self.rollout_length, self.num_envs, *self.obs_shape), device=default_device)
        self.logp_old   = torch.zeros((self.rollout_length, self.num_envs), device=default_device)
        self.value_old  = torch.zeros((self.rollout_length, self.num_envs), device=default_device)
        self.advantages = torch.zeros_like(self.rewards)
        self.returns    = torch.zeros_like(self.rewards)

    def push(self, state, action, reward, next_state, done, logp, value):
        assert self.t < self.rollout_length, f"RolloutBuffer is full (t={self.t})"

        state      = torch.as_tensor(state, dtype=torch.float32)
        action     = torch.as_tensor(action, dtype=torch.float32)
        reward     = torch.as_tensor(reward, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        done       = torch.as_tensor(done, dtype=torch.bool)
        logp       = torch.as_tensor(logp, dtype=torch.float32)
        value      = torch.as_tensor(value, dtype=torch.float32)

        self.states[self.t].copy_(state)
        self.actions[self.t].copy_(action)
        self.rewards[self.t].copy_(reward)
        self.next_states[self.t].copy_(next_state)
        self.dones[self.t].copy_(done)
        self.logp_old[self.t].copy_(logp)
        self.value_old[self.t].copy_(value)
        self.t += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95, normalize_advantages: bool = True):
        last_value = last_value.detach().cpu()
        N, E = self.rollout_length, self.num_envs
        adv = torch.zeros(E)
        for t in reversed(range(N)):
            next_val = last_value if t == N - 1 else self.value_old[t + 1]
            mask = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_val * mask - self.value_old[t]
            adv = delta + gamma * lam * adv * mask
            self.advantages[t] = adv
        self.returns = self.advantages + self.value_old
        if normalize_advantages:
            adv_flat = self.advantages.view(-1)
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            self.advantages = adv_flat.view(N, E)

    def get_batches(self, batch_size: int):
        N, E = self.rollout_length, self.num_envs
        total = N * E
        states   = self.states.permute(1, 0, 2, 3, 4).reshape(total, *self.obs_shape)
        actions  = self.actions.view(total, *self.action_shape)
        logp_old = self.logp_old.view(total)
        returns  = self.returns.view(total)
        advs     = self.advantages.view(total)
        inds = torch.randperm(total)
        for i in range(0, total, batch_size):
            idx = inds[i : i + batch_size]
            yield states[idx], actions[idx], logp_old[idx], returns[idx], advs[idx]

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
