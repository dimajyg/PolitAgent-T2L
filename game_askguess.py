from askguess.game import AskGuessGame
from askguess.utils.utils import get_model
import argparse
import json
import os
import vthread
from time import sleep

@vthread.pool(1)
def run(word, i, model, args):
    game = AskGuessGame(args, model)
    game.init_game(word)
    
    log_dir = f"askguess/logs_{args.mode}_{args.model_name}/{word}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    with open(f"{log_dir}/{i}.log", "w") as f:
        while True:
            result = game.game_loop(f)
            sleep(2)
            if result is not None:
                result["log"] = f"{i}.log"
                break
                
    with open(f"askguess/guess_result_{args.mode}_{args.model_name}.json", "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='askguess/test_labels.json')
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--mode', type=str, default='hard')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    
    model = get_model(args.model_name)
    
    with open(args.label_path, 'r') as f:
        labels = json.load(f)

    for label in labels:
        log_dir = f"askguess/logs_{args.mode}_{args.model_name}/{label}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    for i in range(len(labels)):
        label = labels[i]
        for j in range(args.n):
            run(label, j, model, args)

    vthread.pool.wait()

