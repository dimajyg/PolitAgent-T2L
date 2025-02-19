from beast.game import BeastGame
from beast.utils.utils import get_model
import argparse
import json
import os
import vthread

@vthread.pool(1)
def run(i, model, args):
    game = BeastGame(args, model)
    settings = game.init_game()
    
    log_dir = f"beast/logs/{args.model_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    with open(f"{log_dir}/{i}.log", "w") as f:
        f.write(settings)
        while True:
            result = game.game_loop(f)
            if result is not None:
                break
                
    with open(f"beast/result_{args.model_name}.json", "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    
    model = get_model(args.model_name)

    for i in range(args.n):
        run(i, model, args)

    vthread.pool.wait()