from src.model import DuelDQN
from src.buffer import ReplayBuffer
import gymnasium as gym
import ale_py
import cv2
import yaml
import torch
from torch.optim import Adam
import os
from tqdm import tqdm
import random
import numpy as np
from collections import deque

gym.register_envs(ale_py)

class GameEnv:
    def __init__(self, config: str, env_name: str, weight_path: str):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)

        self.env = gym.make(env_name, obs_type="grayscale")
        self.acs = self.env.action_space.n
        self.obs = self.env.observation_space.shape[0]
        self.env_name = env_name
        
        self.model = DuelDQN(in_channels=self.config.get("frame_stack", 4), output_dim=self.acs)
        self.target_model = DuelDQN(in_channels=self.config.get("frame_stack", 4), output_dim=self.acs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.opt = Adam(self.model.parameters(), lr=self.config["lr"])
        
        self.buffer = ReplayBuffer(memory=self.config["memory"])
        
        self.num_stack = self.config.get("frame_stack", 4)
        self.frame_stack = deque(maxlen=self.num_stack)
        
        if weight_path: 
            self.model.load_weights(weight_path)
            self.target_model.load_weights(weight_path)
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.episodes = self.config["episodes"]
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        
        self.update_target(hard_update=True)
        
    def update_target(self, hard_update: bool=True, tau: float = 0.005):
        if hard_update: 
            self.target_model.load_state_dict(self.model.state_dict())
        else: 
            for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * model_param.data + (1 - tau) * target_param.data)
                
    def preprocess(self, frame):
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        return resized.astype(np.float32)
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        pbar = tqdm(range(self.episodes), desc="Episodes: ")
        
        max_reward = -1e6
        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        
        for i in pbar:
            obs, _ = self.env.reset()
            done = False
            frame = self.preprocess(obs)
            self.frame_stack.clear()
            for _ in range(self.num_stack):
                self.frame_stack.append(frame)
                
            total_reward = 0.0
            total_loss = 0.0
            step = 0

            state = np.stack(self.frame_stack, axis=0)

            epsilon = self.epsilon if i == 0 else max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.epsilon = epsilon
            
            while not done: 
                if self.epsilon > random.random():
                    action = self.env.action_space.sample()
                else: 
                    state_tensor = torch.as_tensor(state, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                    q_values = self.model(state_tensor)
                    action = torch.argmax(q_values)
                    
                obs, reward, terminated, truncated, info = self.env.step(action)
                frame = self.preprocess(obs)
                self.frame_stack.append(frame)
                next_state = np.stack(self.frame_stack, axis=0)
                
                self.buffer.push(
                    state=state, 
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done = terminated or truncated
                )
                
                total_reward += reward
                
                if len(self.buffer) > self.batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    with torch.no_grad():
                        target_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
                        targets = rewards + self.config["gamma"] * target_q_values * (1 - dones)
                    
                    q_values = self.model(states).gather(1, actions)
                    
                    loss = torch.nn.functional.mse_loss(q_values, targets)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    total_loss += loss.item()

                    if step % self.update_freq == 0:
                        self.update_target(hard_update=False)
                        
            avg_reward.append(total_reward)
            avg_loss.append(total_loss)
            
            mva_reward = sum(avg_reward) / len(avg_reward) if len(avg_reward) > 0 else 0.0
            mva_loss = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0.0
                        
            pbar.set_postfix({
                "reward": mva_reward,
                "avg_loss": mva_loss,
                "epsilon": round(self.epsilon, 3)
            })
            
            if max_reward * 1.10 <= mva_reward:
                self.model.save_weights(os.path.join(path, "ddqn.pth"))
                max_reward = mva_reward

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
