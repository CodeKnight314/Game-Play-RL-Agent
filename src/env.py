from src.model import DuelDQN
from src.buffer import ReplayBufferFast
import gymnasium as gym
import ale_py
import cv2
import yaml
import torch
from torch.optim import RMSprop
import os
from tqdm import tqdm
import random
import numpy as np
from collections import deque
from gymnasium.wrappers import FrameStackObservation

gym.register_envs(ale_py)

class GameEnv:
    def __init__(self, config: str, env_name: str, weight_path: str, num_envs: int, resume_step: int):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)

        self.env = gym.vector.AsyncVectorEnv(
            [lambda: FrameStackObservation(gym.make(env_name, obs_type="grayscale"), stack_size=self.config.get("frame_stack", 4)) 
             for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.acs = 4
        self.obs = self.env.observation_space.shape[0]
        self.env_name = env_name
        self.num_envs = num_envs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = DuelDQN(in_channels=self.config.get("frame_stack", 4), output_dim=self.acs).to(self.device)
        self.target_model = DuelDQN(in_channels=self.config.get("frame_stack", 4), output_dim=self.acs).to(self.device)
        
        self.opt = RMSprop(
            self.model.parameters(),
            lr=self.config["lr"],
            alpha=0.95,
            eps=0.01,
            centered=True
        )
        
        self.buffer = ReplayBufferFast()
        
        self.num_stack = self.config.get("frame_stack", 4)
        self.frame_stacks = [deque(maxlen=self.num_stack) for _ in range(num_envs)]
        
        if weight_path: 
            self.model.load_weights(weight_path)
            self.target_model.load_weights(weight_path)
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        
        self.update_target(hard_update=True)
        
        self.start_step = resume_step
        
    def update_target(self, hard_update: bool=True, tau: float = 0.005):
        if hard_update: 
            self.target_model.load_state_dict(self.model.state_dict())
        else: 
            for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * model_param.data + (1 - tau) * target_param.data)

    def update_noise(self, step: int):
        self.epsilon = max(self.epsilon_min, self.config["epsilon"] - (step / 1000000) * (self.config["epsilon"] - self.epsilon_min))
                
    def preprocess(self, frame):
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        return resized.astype(np.float32)
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        pbar = tqdm(initial=self.start_step, total=self.config["max_frames"], desc="Frames: ")
        episode_rewards = [0.0] * self.num_envs
        step = self.start_step
        max_reward = -1e6
        
        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        
        obs, _ = self.env.reset()
        while step < self.config["max_frames"]:
            if random.random() < self.epsilon:
                actions = self.env.action_space.sample()
            else: 
                with torch.no_grad():
                    resized_obs = np.array([
                        [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in env_obs]
                        for env_obs in obs
                    ])
                    state_tensor = torch.FloatTensor(resized_obs).to(self.device)
                    q_values = self.model(state_tensor)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            resized_obs = np.array([
                [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in env_obs]
                for env_obs in obs
            ])

            resized_next_obs = np.array([
                [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in env_obs]
                for env_obs in next_obs
            ])
            
            for i in range(self.num_envs):
                self.buffer.push(resized_obs[i], actions[i], rewards[i], resized_next_obs[i], dones[i])
                episode_rewards[i] += rewards[i]
                
                if dones[i]:
                    avg_reward.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                
            obs = next_obs
        
            if len(self.buffer) > self.batch_size and step % self.config.get("train_freq", 4) == 0:
                states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                current_q_values = self.model(states).gather(1, actions)
                
                with torch.no_grad():
                    next_actions = self.model(next_states).argmax(1, keepdim=True)
                    max_next_q = self.target_model(next_states).gather(1, next_actions)
                    target_q_values = rewards + (1 - dones) * self.config["gamma"] * max_next_q
                    
                current_q_values = self.model(states).gather(1, actions)
                loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
                avg_loss.append(loss.item())
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            
            if step % self.config["update_freq"] == 0:
                tau = self.config.get("tau", 0.005)
                self.update_target(hard_update=False, tau=tau)
            
            self.update_noise(step)
            
            pbar.update(self.num_envs)
            step += self.num_envs
            
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
                epsilon=f"{self.epsilon:.3f}"
            )
            
            if avg_reward_val > max_reward:
                max_reward = avg_reward_val
                self.model.save_weights(os.path.join(path, "model.pth"))
        
        self.model.save_weights(os.path.join(path, "final.pth"))
        
    def test(self, path: str):
        self.env = gym.make(self.env_name, obs_type="grayscale", render_mode="rgb_array")
        self.model.eval()
        os.makedirs(path, exist_ok=True)

        video_path = os.path.join(path, "simulation.mp4")

        obs, _ = self.env.reset()
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = self.preprocess(gray)

        height, width = obs.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        frames = deque(maxlen=self.config["frame_stack"])
        for _ in range(self.config["frame_stack"]):
            frames.append(frame)

        total_reward = 0
        done = False

        while not done:
            state = np.stack(frames, axis=0)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            frame = self.preprocess(gray)
            frames.append(frame)

            rendered_rgb = self.env.render()
            frame_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

            total_reward += reward

        video.release()
        print(f"MP4 video saved to {video_path}")
        print(f"Test completed with total reward: {total_reward}")
        return total_reward
