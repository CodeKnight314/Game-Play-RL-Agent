import argparse
from src.env import GameEnv

def main(args):
    env = GameEnv(args.c, args.env, args.w)
    if args.train:
        env.train(args.o)
        env.test(args.o)
    else: 
        env.test(args.o)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="breakout", choices=["breakout", "pong"], help="Environment for GameEnv")
    parser.add_argument("--c", type=str, required=True, help="config file path")
    parser.add_argument("--w", type=str, help="path to model weights")
    parser.add_argument("--o", type=str, required=True, help="output path to save weights and simulations")
    parser.add_argument("--train", action="store_true", help="flag to toggle training")
    
    args = parser.parse_args()