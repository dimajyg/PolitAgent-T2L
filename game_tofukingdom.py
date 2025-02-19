from tofukingdom.game import TofuKingdomGame
from tofukingdom.utils.utils import get_model
import argparse
import json
import os
import vthread

@vthread.pool(1)
def run(i, prince_model, queen_model, spy_model, args):
    game = TofuKingdomGame(args, prince_model, queen_model, spy_model)
    settings = game.init_game()
    
    log_dir = f"tofukingdom/logs/{args.prince_model_name}_{args.queen_model_name}_{args.spy_model_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    with open(f"{log_dir}/{i}.log", "w") as f:
        f.write(settings)
        while True:
            result = game.game_loop(f)
            if result is not None:
                break
                
    with open(f"tofukingdom/result.json", "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prince_model_name', type=str, default='mistral')
    parser.add_argument('--spy_model_name', type=str, default='mistral')
    parser.add_argument('--queen_model_name', type=str, default='mistral')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    
    prince_model = get_model(args.prince_model_name)
    spy_model = get_model(args.spy_model_name)
    queen_model = get_model(args.queen_model_name)

    for i in range(args.n):
        run(i, prince_model, queen_model, spy_model, args)

    vthread.pool.wait()

            