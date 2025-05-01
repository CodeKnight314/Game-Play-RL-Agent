from src.model import DuelDQN
from src.buffer import ReplayBuffer
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

gym.register_envs(ale_py)

class GameEnv:
    def __init__(self, config: str, env_name: str, weight_path: str, num_envs: int):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)

        self.env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(env_name, obs_type="grayscale") for i in range(num_envs)], 
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
        
        self.buffer = ReplayBuffer(memory=self.config["memory"])
        
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
        
    def update_target(self, hard_update: bool=True, tau: float = 0.005):
        if hard_update: 
            self.target_model.load_state_dict(self.model.state_dict())
        else: 
            for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * model_param.data + (1 - tau) * target_param.data)

    def update_noise(self, step: int):
        self.td3_exploration = max(self.epsilon_min, self.config["epsilon"] - (step / 1000000) * (self.config["epsilon"] - self.epsilon_min))
                
    def preprocess(self, frame):
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        return resized.astype(np.float32)
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        pbar = tqdm(total=self.config["max_frames"], desc="Frames: ")

        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        
        # Initialize frame stacks for each environment
        obs, _ = self.env.reset()
        for i in range(self.num_envs):
            frame = self.preprocess(obs[i])
            self.frame_stacks[i] = deque(maxlen=self.num_stack)
            for _ in range(self.num_stack):
                self.frame_stacks[i].append(frame)

        episode_rewards = [0.0] * self.num_envs
        step = 0
        max_reward = -1e6
        
        while step < int(self.config["max_frames"]):
            # Update epsilon with linear decay
            self.epsilon = max(self.epsilon_min, self.config["epsilon"] - (step / self.config["max_frames"]) * 
                            (self.config["epsilon"] - self.epsilon_min))
            
            # Select actions (vectorized)
            if np.random.random() < self.epsilon:
                actions = np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                state_tensors = torch.as_tensor(np.array([np.stack(fs, axis=0) for fs in self.frame_stacks]), 
                                            dtype=torch.float32).to(self.device)
                q_values = self.model(state_tensors)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            # Execute actions in all environments
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # Process each environment
            for i in range(self.num_envs):
                # Current state (before step)
                curr_state = np.stack(self.frame_stacks[i], axis=0)
                
                # Process new observation
                frame = self.preprocess(next_obs[i])
                self.frame_stacks[i].append(frame)
                
                # Next state (after adding new frame)
                next_state = np.stack(self.frame_stacks[i], axis=0)
                
                # Store transition in replay buffer
                self.buffer.push(
                    curr_state, 
                    actions[i],
                    rewards[i],
                    next_state, 
                    terminateds[i] or truncateds[i]
                )
                
                # Update episode rewards
                episode_rewards[i] += rewards[i]
                
                # Check if episode is done
                if terminateds[i] or truncateds[i]:
                    avg_reward.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
            
            # Update step counter and progress bar
            step += self.num_envs
            pbar.update(self.num_envs)
            
            # Training step
            if len(self.buffer) > 10000:
                states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                
                # Double DQN update
                with torch.no_grad():
                    next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
                    target_q_values = self.target_model(next_states).gather(1, next_actions)
                    targets = rewards + self.config["gamma"] * target_q_values * (1 - dones)
                
                # Current Q-values
                q_values = self.model(states).gather(1, actions)
                
                # Compute loss and update
                loss = torch.nn.functional.mse_loss(q_values, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                # Record loss
                avg_loss.append(loss.item())
                
                # Update target network periodically
                if step % self.update_freq == 0:
                    self.update_target(hard_update=True)  # Using hard update as in original paper
            
            # Update progress bar
            mva_reward = sum(avg_reward) / len(avg_reward) if avg_reward else 0.0
            mva_loss = sum(avg_loss) / len(avg_loss) if avg_loss else 0.0
            
            pbar.set_postfix({
                "reward": round(mva_reward, 2),
                "loss": round(mva_loss, 5),
                "epsilon": round(self.epsilon, 3)
            })
            
            # Save model on improvement
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
