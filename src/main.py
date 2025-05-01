import argparse
from src.env import GameEnv

MODEL_ENV_MAP = {
    "breakout": "ALE/Breakout-v5",
    "pong":"ALE/Pong-v5"
}

def main(args):
    env_name = MODEL_ENV_MAP.get(args.model)
    if env_name is None:
        raise ValueError(f"Invalid model type '{args.model}'. Choose from: {', '.join(MODEL_ENV_MAP.keys())}")
    
    env = GameEnv(args.c, env_name, args.w)
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
    
    main(args)