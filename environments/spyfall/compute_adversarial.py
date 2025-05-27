import json
import os
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_PATH = "environments/spyfall/prompts/labels.txt"
LOGS_DIR = "environments/spyfall/logs/gpt3_gpt4"
OUT_PATH = "environments/spyfall/results/gpt3_gpt4.json"

def read_labels(path: str) -> List[List[str]]:
    with open(path, 'r') as f:
        return [line.strip().split(",") for line in f]

def compute_adversarial(lines: List[str]) -> Dict[str, Any]:
    cnt_spy = cnt_villager = round_sum = 0
    for line in lines:
        item = json.loads(line)
        winner = item.get("winner") or item.get("winer")
        if winner == "exit":
            continue
        if winner == "spy":
            cnt_spy += 1
        else:
            cnt_villager += 1
        round_sum += item["round"]
    total = cnt_spy + cnt_villager
    round_avg = round_sum / total if total else 0
    rate = cnt_spy / total if total else 0
    return {"cnt_spy": cnt_spy, "cnt_villager": cnt_villager, "round_avg": round_avg, "rate": rate}

def main() -> None:
    labels = read_labels(LABELS_PATH)
    res_dict = {}
    for label in labels:
        label_name = f"{label[0]}&{label[1]}"
        dir_name = os.path.join(LOGS_DIR, label_name)
        res_file = os.path.join(dir_name, "res.json")
        if not os.path.exists(res_file):
            logger.warning(f"File not found: {res_file}")
            continue
        with open(res_file, 'r') as f:
            lines = f.readlines()
        res = compute_adversarial(lines)
        res_dict[label_name] = res

    valid_results = [v for k, v in res_dict.items() if k != "avg"]
    avg = {
        "round_avg": sum(v["round_avg"] for v in valid_results) / len(valid_results) if valid_results else 0,
        "rate": sum(v["rate"] for v in valid_results) / len(valid_results) if valid_results else 0,
    }
    res_dict["avg"] = avg

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(res_dict, f, indent=2)
    logger.info(f"Results written to {OUT_PATH}")

if __name__ == "__main__":
    main()
    