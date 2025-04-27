from spyfall.game import SpyfallGame
from spyfall.utils.utils import get_model
from spyfall.metrics.evaluation import SpyfallMetrics
import argparse
import json
import os
import vthread
import random
import numpy as np

# Store results for metrics calculation
all_results = []

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
    
    # Add to global results
    all_results.append(result)
                
    with open(f"spyfall/result_{args.spy_model_name}_{args.villager_model_name}.json", "a") as f:
        f.write(json.dumps(result) + "\n")

def calculate_metrics(results, openai_api_key=None, embedding_model="local", 
                      embedding_model_name="all-MiniLM-L6-v2", perplexity_model="local"):
    """
    Calculate metrics for game results.
    
    Args:
        results: List of game result dictionaries
        openai_api_key: OpenAI API key (overrides env variable)
        embedding_model: "local" or "openai"
        embedding_model_name: Name of embedding model
        perplexity_model: Model to use for perplexity calculation
        
    Returns:
        Dictionary with calculated metrics
    """
    # Используем API ключ из аргументов или переменной окружения
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Выводим информацию о выбранных моделях
    print(f"Embedding model: {embedding_model} ({embedding_model_name})")
    print(f"Perplexity model: {perplexity_model}")
    print(f"OpenAI API key available: {bool(openai_api_key)}")
    
    # Get all descriptions across games
    all_descriptions = []
    for result in results:
        if "descriptions" in result:
            all_descriptions.extend(result["descriptions"].values())
            
    # Calculate vagueness for all descriptions together
    vagueness_scores = SpyfallMetrics.vagueness_score(all_descriptions) if all_descriptions else {}
    
    # Calculate perplexity for each description using the specified model
    if hasattr(SpyfallMetrics, "calculate_perplexity"):
        for result in results:
            if "description_metrics" in result:
                for player, metrics_data in result["description_metrics"].items():
                    description = result.get("descriptions", {}).get(player, "")
                    if description:
                        metrics_data["perplexity"] = SpyfallMetrics.calculate_perplexity(
                            description, 
                            api_key=openai_api_key,
                            model_name=perplexity_model
                        )
            
    # Calculate CoT coherence using the specified embedding model
    cot_coherence = SpyfallMetrics.cot_coherence(
        results, 
        model_type=embedding_model,
        model_name=embedding_model_name,
        api_key=openai_api_key
    )
            
    metrics = {
        "spy_survival_rate": SpyfallMetrics.spy_survival_rate(results),
        "villager_detection": SpyfallMetrics.villager_detection_metrics(results),
        "matthews_correlation": SpyfallMetrics.matthews_correlation(results),
        "top_3_suspect_accuracy": SpyfallMetrics.top_k_suspect_accuracy(results, k=3),
        "avg_description_specificity": {
            "all": np.mean([
                d["specificity"] for r in results if "description_metrics" in r 
                for d in r["description_metrics"].values()
            ]) if any("description_metrics" in r for r in results) else 0.0,
            "spy": np.mean([
                r["description_metrics"][r["players"][r["spy_index"]-1]]["specificity"]
                for r in results if "description_metrics" in r and "spy_index" in r
            ]) if any("description_metrics" in r for r in results) else 0.0,
            "villagers": np.mean([
                d["specificity"] for r in results if "description_metrics" in r 
                for p, d in r["description_metrics"].items() 
                if p != r["players"][r["spy_index"]-1]
            ]) if any("description_metrics" in r for r in results) else 0.0
        },
        "avg_perplexity": {
            "all": np.mean([
                d["perplexity"] for r in results if "description_metrics" in r 
                for d in r["description_metrics"].values()
            ]) if any("description_metrics" in r for r in results) else 0.0,
            "spy": np.mean([
                r["description_metrics"][r["players"][r["spy_index"]-1]]["perplexity"]
                for r in results if "description_metrics" in r and "spy_index" in r
            ]) if any("description_metrics" in r for r in results) else 0.0,
            "villagers": np.mean([
                d["perplexity"] for r in results if "description_metrics" in r 
                for p, d in r["description_metrics"].items() 
                if p != r["players"][r["spy_index"]-1]
            ]) if any("description_metrics" in r for r in results) else 0.0
        },
        "avg_vagueness": {
            "all": np.mean(list(vagueness_scores.values())) if vagueness_scores else 0.0,
            # Other vagueness metrics would be calculated similarly
        },
        "brier_score": SpyfallMetrics.brier_score(results),
        "vote_influence_index": SpyfallMetrics.vote_influence_index(results),
        "cot_coherence": cot_coherence,
        "metrics_calculation": {
            "embedding_model": f"{embedding_model}:{embedding_model_name}",
            "perplexity_model": perplexity_model
        }
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='spyfall/labels.txt', help='path to word pairs for game')
    parser.add_argument('--spy_model_name', type=str, default='openai', help='model name for spy')
    parser.add_argument('--villager_model_name', type=str, default='openai', help='model name for villagers')
    parser.add_argument('--n', type=int, default=1, help='number of games')
    parser.add_argument('--debug', type=bool, default=True, help='debug mode')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key (overrides env variable)')
    parser.add_argument('--embedding_model', type=str, default='auto', choices=['local', 'openai', 'auto'], 
                        help='Type of embedding model to use for coherence metric')
    parser.add_argument('--embedding_model_name', type=str, default='text-embedding-3-large', 
                        help='Name of embedding model')
    parser.add_argument('--perplexity_model', type=str, default='auto', 
                        help='Model to use for perplexity calculation (auto/local/model name)')
    
    args = parser.parse_args()
    
    # Автоматически определяем тип модели, если указано auto
    if args.embedding_model == 'auto':
        # Если есть OPENAI_API_KEY, используем OpenAI
        if os.environ.get("OPENAI_API_KEY") or args.openai_api_key:
            args.embedding_model = 'openai'
        else:
            args.embedding_model = 'local'
            
    if args.perplexity_model == 'auto':
        # Если есть OPENAI_API_KEY, используем gpt-3.5-turbo-instruct
        if os.environ.get("OPENAI_API_KEY") or args.openai_api_key:
            args.perplexity_model = 'gpt-3.5-turbo-instruct'
        else:
            args.perplexity_model = 'local'
    
    with open(args.label_path, 'r') as f:
        data = f.readlines()

    labels = []
    for item in data:
        labels.append(item.strip().split(","))

    spy_model = get_model(args.spy_model_name)
    villager_model = get_model(args.villager_model_name)

    for j in range(args.n):
        idx_pair = random.randint(0, len(labels) - 1)
        label = labels[idx_pair]
        run(label, j, spy_model, villager_model, args)

    vthread.pool.wait()
    
    # Calculate and save metrics with model selection
    metrics = calculate_metrics(
        all_results, 
        openai_api_key=args.openai_api_key,
        embedding_model=args.embedding_model,
        embedding_model_name=args.embedding_model_name,
        perplexity_model=args.perplexity_model
    )
    metrics_path = f"spyfall/metrics/{args.spy_model_name}_{args.villager_model_name}_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics Summary:")
    print(f"Spy Survival Rate: {metrics['spy_survival_rate']:.2f}")
    print(f"Villager Detection F1: {metrics['villager_detection']['f1']:.2f}")
    print(f"Matthews Correlation Coefficient: {metrics['matthews_correlation']:.2f}")
    print(f"Top-3 Suspect Accuracy: {metrics['top_3_suspect_accuracy']:.2f}")
    print(f"Description Specificity (All/Spy/Villagers): {metrics['avg_description_specificity']['all']:.2f}/{metrics['avg_description_specificity']['spy']:.2f}/{metrics['avg_description_specificity']['villagers']:.2f}")
    print(f"Perplexity (All/Spy/Villagers): {metrics['avg_perplexity']['all']:.2f}/{metrics['avg_perplexity']['spy']:.2f}/{metrics['avg_perplexity']['villagers']:.2f}")
    print(f"Vagueness Score (All): {metrics['avg_vagueness']['all']:.2f}")
    print(f"Brier Score: {metrics['brier_score']:.2f} (lower is better)")
    print("Vote Influence by Position:", end=" ")
    for pos, score in sorted(metrics['vote_influence_index'].items(), key=lambda x: int(x[0])):
        print(f"{pos}:{score:.2f}", end=" ")
    print()
    print(f"CoT Coherence (All/Spy/Villagers): {metrics['cot_coherence']['all']:.2f}/{metrics['cot_coherence']['spy']:.2f}/{metrics['cot_coherence']['villagers']:.2f}")
    print("Human Clarity: Manual evaluation required")