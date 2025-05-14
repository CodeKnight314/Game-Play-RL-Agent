from src.model import ActorCritc
from src.buffer import RolloutBuffer
import gymnasium as gym
import ale_py
import cv2
import yaml
import torch
from torch.optim import Adam
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from collections import deque
from gymnasium.wrappers import FrameStackObservation

gym.register_envs(ale_py)

class GameEnv:
    def __init__(self, config: str, env_name: str, weight_path: str, num_envs: int, resume_step: int = 0):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: FrameStackObservation(gym.make(env_name, obs_type="grayscale"), stack_size=self.config.get("frame_stack", 4))
             for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.num_actions = self.env.action_space[0].n
        
        self.obs_shape = (self.config.get("frame_stack", 4), 84, 84)
        self.env_name = env_name 
        self.num_envs = num_envs
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = ActorCritc(
            input_shape=self.obs_shape, 
            hidden_dim=self.config.get("hidden_dim", 512), 
            num_actions=self.num_actions
        ).to(self.device)
        
        self.opt = Adam(
            self.model.parameters(), 
            lr=self.config["lr"]
        )
        
        self.rollout_length = self.config.get("rollout_length", 128)
        self.buffer = RolloutBuffer(
            rollout=self.rollout_length, 
            num_envs=num_envs,
            obs_shape=self.obs_shape,
            action_shape=(), 
        )
        
        if weight_path: 
            self.model.load_state_dict(torch.load(weight_path))
            
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.ppo_epochs = self.config.get("ppo_epochs", 4)
        self.batch_size = self.config.get("batch_size", 256)
        
        self.start_step = resume_step
        
    def preprocess_obs(self, obs: np.array):
        resized_obs = np.array([
            [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in env_obs]
            for env_obs in obs
        ])
        
        normalized_obs = resized_obs / 255.0
    
        return torch.FloatTensor(normalized_obs).to(self.device)
    
    def collect_rollouts(self):
        self.buffer.reset()
        obs, _ = self.env.reset()
        
        states = self.preprocess_obs(obs)

        total_rewards = np.zeros((self.num_envs))
        
        for step in range(self.rollout_length):
            with torch.no_grad():
                actions, log_probs, values = self.model.get_action(states)
                
            actions_np = actions.cpu().numpy()
            
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)
            
            next_states = self.preprocess_obs(next_obs)
            
            self.buffer.push(
                states.cpu(),
                actions.cpu(),
                rewards, 
                next_states.cpu(),
                dones, 
                log_probs.cpu(),
                values.cpu()
            )
            
            obs = next_obs
            states = next_states
            
            if step == 0:
                total_rewards = rewards
            else: 
                total_rewards = np.concatenate([total_rewards, rewards]).reshape(-1)
            
        with torch.no_grad(): 
            _, _, last_value = self.model.get_action(states)
            
        self.buffer.compute_gae(
            last_value=last_value.cpu(),
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        
        self.buffer.to(self.device)
        
        return np.sum(total_rewards)
        
    def update_policy(self):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        for epoch in range(self.ppo_epochs):
            for states, actions, old_log_probs, returns, advantages in self.buffer.get_batches(self.batch_size):
                log_probs, values, entropy = self.model.evaluate_actions(states, actions)
                values = values.squeeze(-1)
                
                ratio = torch.exp(log_probs - old_log_probs)
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                value_loss = F.mse_loss(values, returns)
                
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
        num_updates = self.ppo_epochs * (self.rollout_length * self.num_envs // self.batch_size + 1)
        avg_loss = total_loss / num_updates
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy_loss
    
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        pbar = tqdm(initial=self.start_step, total=self.config["max_frames"], desc="Frames: ")
        step = self.start_step
        max_reward = -1e6
        
        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        
        while step < self.config["max_frames"]:
            total_rewards = self.collect_rollouts()
            avg_reward.append(total_rewards)
            
            frames_in_rollout = self.rollout_length * self.num_envs
            pbar.update(frames_in_rollout)
            step += frames_in_rollout
            
            loss, policy_loss, value_loss, entropy_loss = self.update_policy()
            avg_loss.append(loss)
            
            if avg_reward:
                avg_reward_val = sum(avg_reward) / len(avg_reward)
            else:
                avg_reward_val = 0.0
                
            if avg_loss:
                avg_loss_val = sum(avg_loss) / len(avg_loss)
            else:
                avg_loss_val = 0.0
            
            pbar.set_postfix(
                reward=f"{avg_reward_val:.3f}",
                loss=f"{avg_loss_val:.3f}",
                policy_loss=f"{policy_loss:.3f}",
                value_loss=f"{value_loss:.3f}",
                entropy=f"{entropy_loss:.3f}",
            )
            
            if avg_reward_val > max_reward:
                max_reward = avg_reward_val
                torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        
        torch.save(self.model.state_dict(), os.path.join(path, "final.pth"))
        
    def test(self, path: str):
        os.makedirs(path, exist_ok=True)
        pass