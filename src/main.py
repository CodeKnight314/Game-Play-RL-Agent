import argparse
import os
from src.env import PPOEnv, DQNEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents on Atari games')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'dqn'], help='Algorithm to use')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5', help='Environment name')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume_step', type=int, default=0, help='Step to resume from')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run in')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    results_dir = os.path.join('results', args.algo)
    os.makedirs(results_dir, exist_ok=True)
    
    if args.algo == 'ppo':
        env = PPOEnv(
            config=args.config,
            env_name=args.env,
            weight_path=args.resume,
            num_envs=args.num_envs,
            resume_step=args.resume_step
        )
    else:
        env = DQNEnv(
            config=args.config,
            env_name=args.env,
            weight_path=args.resume,
            num_envs=args.num_envs,
            resume_step=args.resume_step
        )
    

    if args.mode == 'train':
        env.train(results_dir)
        env.plot_history(results_dir)
    else:
        env.test(results_dir)

if __name__ == '__main__':
    main() 