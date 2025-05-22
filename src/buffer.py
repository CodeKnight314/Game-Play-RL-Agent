from collections import deque
import torch
import random 
from typing import Tuple
    
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

class ReplayBuffer: 
    def __init__(self, max: int = 10000):
        self.memory = deque(maxlen=max)
        
    def push(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).detach()
        actions = torch.stack(actions).to(actions[0].device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self): 
        return len(self.memory)
    