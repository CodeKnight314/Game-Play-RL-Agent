from src.model import ActorCritc, DQN
from src.buffer import RolloutBuffer, ReplayBuffer
import gymnasium as gym
import ale_py
import cv2
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from collections import deque
from gymnasium.wrappers import FrameStackObservation
import matplotlib.pyplot as plt
import math

gym.register_envs(ale_py)

# Borrowed from https://github.com/DarylRodrigo/rl_lib/blob/master/PPO/envs.py
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        next_obs, rewards, terminateds, truncateds, infos = self.env.step(action)
        self.was_real_done = terminateds or truncateds
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminateds = True
            truncated = True
        self.lives = lives
        return next_obs, rewards, terminateds, truncateds, infos

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)

class PPOEnv:
    def __init__(self, config: str, env_name: str="ALE/Pong-v5", weight_path: str=None, num_envs: int=1, resume_step: int = 0):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: self._make_env(env_name) for i in range(num_envs)], 
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
        
        self.max_frames = self.config["max_frames"]
        self.lr_min = self.config.get("min_lr", 0.00001)  # Minimum learning rate
        self.scheduler = CosineAnnealingLR(
            self.opt, 
            T_max=self.max_frames // (self.num_envs * self.config.get("rollout_length", 128)),  # Convert frames to updates
            eta_min=self.lr_min
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
            print(f"[INFO] Loaded model from {weight_path}")

        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.base_entropy_coef = self.config["entropy_coef"]
        self.e_decay_end = self.config.get("e_decay_end", 1_000_000)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.ppo_epochs = self.config.get("ppo_epochs", 4)
        self.batch_size = self.config.get("batch_size", 256)
        
        self.start_step = resume_step

    def _make_env(self, env_name: str):
        env = gym.make(env_name, obs_type="grayscale")
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        env = FrameStackObservation(env, stack_size=self.config.get("frame_stack", 4))
        return env
        
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

        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        completed_episodes = 0
        total_episode_rewards = 0.0
        
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
            
            for i in range(self.num_envs):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                
                if dones[i]:
                    completed_episodes += 1
                    total_episode_rewards += episode_rewards[i]
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
        with torch.no_grad(): 
            _, _, last_value = self.model.get_action(states)
            
        self.buffer.compute_gae(
            last_value=last_value.cpu(),
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        
        self.buffer.to(self.device)
        
        avg_episode_reward = total_episode_rewards / completed_episodes if completed_episodes > 0 else 0.0
        avg_episode_length = sum(episode_lengths) / completed_episodes if completed_episodes > 0 else 0
        return avg_episode_reward, completed_episodes, avg_episode_length

    def update_entropy(self, frames: int):
        frac = 1.0 - (frames / self.e_decay_end)
        self.entropy_coef = max(self.base_entropy_coef * frac, 0.03)
        
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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
        total_episodes = 0

        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        avg_episode_length = deque(maxlen=self.config["window_ma"])
        self.reward_history = []
        self.loss_history = []
        
        while step < self.config["max_frames"]:
            avg_episode_reward, completed_episodes, avg_episode_length_value = self.collect_rollouts()
            if completed_episodes > 0:
                avg_reward.append(avg_episode_reward)
                avg_episode_length.append(avg_episode_length_value)
                total_episodes += completed_episodes

            frames_in_rollout = self.rollout_length * self.num_envs
            pbar.update(frames_in_rollout)
            step += frames_in_rollout
            self.update_entropy(step)
            
            loss, policy_loss, value_loss, entropy_loss = self.update_policy()
            self.scheduler.step()
            
            avg_loss.append(loss)
            
            if avg_reward:
                avg_reward_val = sum(avg_reward) / len(avg_reward)
            else:
                avg_reward_val = 0.0
                
            if avg_loss:
                avg_loss_val = sum(avg_loss) / len(avg_loss)
            else:
                avg_loss_val = 0.0

            if avg_episode_length:
                avg_episode_length_val = sum(avg_episode_length) / len(avg_episode_length)
            else: 
                avg_episode_length_val = 0.0

            pbar.set_postfix(
                reward=f"{avg_reward_val:.3f}",
                loss=f"{avg_loss_val:.3f}",
                policy_loss=f"{policy_loss:.3f}",
                value_loss=f"{value_loss:.3f}",
                entropy=f"{(-entropy_loss):.3f} x {(self.entropy_coef):.3f}",
                episodes=f"{total_episodes}",
                length=f"{avg_episode_length_val:.3f}"
            )

            self.reward_history.append(avg_reward_val)
            self.loss_history.append(avg_loss_val)
            
            if avg_reward_val > max_reward:
                max_reward = avg_reward_val
                torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        
        torch.save(self.model.state_dict(), os.path.join(path, "final.pth"))

    def plot_history(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history, label="Reward")
        plt.xlabel("frames")
        plt.ylabel("reward")
        plt.title("Reward History")
        plt.legend()
        plt.savefig(os.path.join(path, "reward_history.png"))
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="Loss")
        plt.xlabel("frames")
        plt.ylabel("loss")
        plt.title("Loss History")
        plt.legend()
        plt.savefig(os.path.join(path, "loss_history.png"))
        
        plt.close()
        
    def test(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        test_env = gym.make(self.env_name, obs_type="grayscale", render_mode="rgb_array")
        test_env = FrameStackObservation(test_env, stack_size=self.config.get("frame_stack", 4))
        
        frames = []
        
        obs, _ = test_env.reset()
        state = torch.FloatTensor(self.preprocess_obs(obs)).unsqueeze(0).to(self.device)
        done = False
        
        total_reward = 0
        step = 0
        
        print("Starting game test...")
        while not done:
            with torch.no_grad():
                action, _, _ = self.model.get_action(state, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = test_env.step(action_np)
            
            frame = test_env.render()
            frames.append(frame)
            
            next_state = torch.FloatTensor(self.preprocess_obs(next_obs)).unsqueeze(0).to(self.device)
            state = next_state
            
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        print(f"Test completed. Total steps: {step}, Total reward: {total_reward}")
        
        try:
            import cv2
            
            video_path = os.path.join(path, "gameplay.mp4")
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            video.release()
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            
            frames_dir = os.path.join(path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(os.path.join(frames_dir, f"frame_{i:05d}.png"), frame)
            print(f"Saved {len(frames)} frames to {frames_dir}")
        
        test_env.close()

    def find_max_batch_size(self) -> int:
        if not torch.cuda.is_available():
            print("CUDA not available, using default batch size of 256")
            return 256

        left = 1
        right = 4096
        max_batch_size = 256
        
        dummy_input = torch.randn(1, *self.obs_shape).to(self.device)
        
        try:
            torch.cuda.empty_cache()
            env_batch = dummy_input.repeat(self.num_envs, 1, 1, 1)
            with torch.no_grad():
                self.model.eval()
                _ = self.model(env_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: Cannot even handle base environment setup with {self.num_envs} environments")
                return 256
            else:
                raise e
        
        while left <= right:
            mid = (left + right) // 2
            try:
                torch.cuda.empty_cache()
                
                total_samples = self.num_envs * mid
                batch_input = dummy_input.repeat(total_samples, 1, 1, 1)
                
                with torch.no_grad():
                    self.model.eval()
                    _ = self.model(batch_input)
                
                max_batch_size = mid
                left = mid + 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = mid - 1
                else:
                    raise e
            except Exception as e:
                print(f"Error during batch size test: {e}")
                break

        max_batch_size = int(max_batch_size * 0.95)
        
        print(f"Determined maximum batch size: {max_batch_size} (with {self.num_envs} environments)")
        return max_batch_size
    

class DQNEnv:
    def __init__(self, config: str, env_name: str="ALE/Pong-v5", weight_path: str=None, num_envs: int=1, resume_step: int = 0):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: self._make_env(env_name) for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.num_actions = self.env.action_space[0].n
        self.obs_shape = (self.config.get("frame_stack", 4), 84, 84)
        self.env_name = env_name 
        self.num_envs = num_envs
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = DQN(
            input_shape=self.obs_shape, 
            hidden_dim=self.config.get("hidden_dim", 512), 
            num_actions=self.num_actions
        ).to(self.device)
        
        self.target_model = DQN(
            input_shape=self.obs_shape, 
            hidden_dim=self.config.get("hidden_dim", 512), 
            num_actions=self.num_actions
        ).to(self.device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.opt = Adam(
            self.model.parameters(), 
            lr=self.config["lr"]
        )
        
        self.buffer = ReplayBuffer(
            capacity=self.config.get("buffer_size", 100000),
            obs_shape=self.obs_shape,
            action_shape=(),
            device=self.device
        )
        
        if weight_path: 
            self.model.load_state_dict(torch.load(weight_path))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"[INFO] Loaded model from {weight_path}")

        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon_start = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.1)
        self.epsilon_decay = self.config.get("epsilon_decay", 1000000)
        self.target_update = self.config.get("target_update", 10000)
        self.batch_size = self.config.get("batch_size", 32)
        self.start_step = resume_step
        
    def _make_env(self, env_name: str):
        env = gym.make(env_name, obs_type="grayscale")
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        env = FrameStackObservation(env, stack_size=self.config.get("frame_stack", 4))
        return env
        
    def preprocess_obs(self, obs: np.array):
        resized_obs = np.array([
            [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in env_obs]
            for env_obs in obs
        ])
        
        normalized_obs = resized_obs / 255.0
    
        return torch.FloatTensor(normalized_obs).to(self.device)
    
    def get_epsilon(self, step):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * step / self.epsilon_decay)
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        with torch.no_grad():
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_model(next_states).max(1)[0].detach()

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        return loss.item()
    
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        pbar = tqdm(initial=self.start_step, total=self.config["max_frames"], desc="Frames: ")
        step = self.start_step
        max_reward = -1e6
        total_episodes = 0

        avg_reward = deque(maxlen=self.config["window_ma"])
        avg_loss = deque(maxlen=self.config["window_ma"])
        self.reward_history = []
        self.loss_history = []
        
        obs, _ = self.env.reset()
        states = self.preprocess_obs(obs)
        
        while step < self.config["max_frames"]:
            epsilon = self.get_epsilon(step)
            actions = self.model.get_action(states, epsilon)
            actions_np = actions.cpu().numpy()
            
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)
            next_states = self.preprocess_obs(next_obs)
            
            for i in range(self.num_envs):
                self.buffer.push(
                    states[i].cpu(),
                    actions[i].cpu(),
                    rewards[i],
                    next_states[i].cpu(),
                    dones[i]
                )
            
            loss = self.train_step()
            if loss > 0:
                avg_loss.append(loss)
            
            if step % self.target_update == 0:
                self.update_target()
            
            obs = next_obs
            states = next_states
            
            for i in range(self.num_envs):
                if dones[i]:
                    total_episodes += 1
                    episode_reward = infos[i].get('episode', {}).get('r', 0)
                    avg_reward.append(episode_reward)
                    self.reward_history.append(episode_reward)
            
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
                epsilon=f"{epsilon:.3f}",
                episodes=f"{total_episodes}"
            )
            
            if avg_reward_val > max_reward:
                max_reward = avg_reward_val
                torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        
        torch.save(self.model.state_dict(), os.path.join(path, "final.pth"))
        
    def plot_history(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history, label="Reward")
        plt.xlabel("frames")
        plt.ylabel("reward")
        plt.title("Reward History")
        plt.legend()
        plt.savefig(os.path.join(path, "reward_history.png"))
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="Loss")
        plt.xlabel("frames")
        plt.ylabel("loss")
        plt.title("Loss History")
        plt.legend()
        plt.savefig(os.path.join(path, "loss_history.png"))
        
        plt.close()
        
    def test(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        test_env = gym.make(self.env_name, obs_type="grayscale", render_mode="rgb_array")
        test_env = FrameStackObservation(test_env, stack_size=self.config.get("frame_stack", 4))
        
        frames = []
        
        obs, _ = test_env.reset()
        state = torch.FloatTensor(self.preprocess_obs(obs)).unsqueeze(0).to(self.device)
        done = False
        
        total_reward = 0
        step = 0
        
        print("Starting game test...")
        while not done:
            with torch.no_grad():
                action = self.model.get_action(state, epsilon=0.0)
            
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = test_env.step(action_np)
            
            frame = test_env.render()
            frames.append(frame)
            
            next_state = torch.FloatTensor(self.preprocess_obs(next_obs)).unsqueeze(0).to(self.device)
            state = next_state
            
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        print(f"Test completed. Total steps: {step}, Total reward: {total_reward}")
        
        try:
            import cv2
            
            video_path = os.path.join(path, "gameplay.mp4")
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            video.release()
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            
            frames_dir = os.path.join(path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(os.path.join(frames_dir, f"frame_{i:05d}.png"), frame)
            print(f"Saved {len(frames)} frames to {frames_dir}")
        
        test_env.close()