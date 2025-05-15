import argparse
from src.env import GameEnv

def main(args):
    env = GameEnv(args.c, "ALE/Pong-v5", args.w, args.num_envs, args.resume_step)
    if args.train:
        try:
            env.train(args.o)
            env.test(args.o)
            env.plot_history(args.o)
        except KeyboardInterrupt:
            env.plot_history(args.o)
    else: 
        env.test(args.o)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, required=True, help="config file path")
    parser.add_argument("--w", type=str, help="path to model weights")
    parser.add_argument("--o", type=str, required=True, help="output path to save weights and simulations")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to run for parallel")
    parser.add_argument("--train", action="store_true", help="flag to toggle training")
    parser.add_argument("--resume_step", type=int, default=0, help="if no checkpoint file is given, start from this global step number")
    args = parser.parse_args()
    
    main(args)