from spyfall.game import SpyfallGame
from spyfall.utils.utils import get_model
import argparse
import json
import os
import vthread


@vthread.pool(1)
def run(phrase_pair, i, spy_model, villager_model, args):
    game = SpyfallGame(args, spy_model, villager_model)
    settings = game.init_game(phrase_pair)
    
    log_dir = f"spyfall/logs/{args.spy_model_name}_{args.villager_model_name}/{phrase_pair[0]}&{phrase_pair[1]}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    with open(f"{log_dir}/{i}.log", "w") as f:
        f.write(settings)
        while True:
            result = game.game_loop(f)
            if result is not None:
                break
                
    with open(f"spyfall/result_{args.spy_model_name}_{args.villager_model_name}.json", "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='spyfall/labels.txt')
    parser.add_argument('--spy_model_name', type=str, default='mistral')
    parser.add_argument('--villager_model_name', type=str, default='mistral')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    
    with open(args.label_path, 'r') as f:
        data = f.readlines()

    labels = []
    for item in data:
        labels.append(item.strip().split(","))

    spy_model = get_model(args.spy_model_name)
    villager_model = get_model(args.villager_model_name)

    for j in range(args.n):
        for i in range(len(labels)):
            label = labels[i]
            run(label, j, spy_model, villager_model, args)

    vthread.pool.wait()