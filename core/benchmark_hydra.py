#!/usr/bin/env python
"""
PolitAgent Benchmark with Hydra - improved interface for managing experiments
with configuration management and experiment tracking.
"""

import os
import json
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from llm.models import get_model, get_available_models
from core.benchmark import GAME_ENVIRONMENTS, run_game

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_hydra")

class HydraExperimentTracker:
    """Tracks experiments, logs configurations, and saves results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []
        self.start_time = datetime.now()
        
    def log_experiment_config(self, config: DictConfig) -> None:
        """Logs experiment configuration."""
        config_path = Path(self.output_dir) / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(config=config, f=f)
            
    def log_result(self, result: Dict[str, Any]) -> None:
        """Logs a single experiment result."""
        self.results.append(result)
        
        result_dir = Path(self.output_dir) / result["game_type"]
        result_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = result_dir / f"{result['timestamp']}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
            
    def save_summary(self) -> None:
        """Saves experiment summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            "total_experiments": len(self.results),
            "duration": str(duration),
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "results": self.results
        }
        
        summary_path = Path(self.output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Experiment summary saved to {summary_path}")

def setup_model_from_config(config: DictConfig) -> Dict[str, Any]:
    """Sets up model based on configuration."""
    model_name = config.model.name
    model_kwargs = {}
    
    if hasattr(config.model, "specific_model"):
        model_kwargs["specific_model"] = config.model.specific_model
        
    if model_name == "ollama" and hasattr(config.model, "ollama_base_url"):
        model_kwargs["base_url"] = config.model.ollama_base_url
        
    return get_model(model_name, **model_kwargs)

def prepare_game_configs_from_hydra(config: DictConfig) -> List[Dict[str, Any]]:
    """Prepares game configurations from Hydra config."""
    game_configs = []
    
    for game_name in config.games:
        if game_name not in GAME_ENVIRONMENTS:
            logger.error(f"Unknown game: {game_name}")
            continue
            
        game_info = GAME_ENVIRONMENTS[game_name]
        game_config = {
            "game_type": game_name,
            "args": game_info["default_args"].copy()
        }
        
        if hasattr(config, game_name):
            game_specific_config = getattr(config, game_name)
            for key, value in game_specific_config.items():
                if key in game_config["args"]:
                    game_config["args"][key] = value
                    
        game_configs.append(game_config)
        
    return game_configs

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point for Hydra-based benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmark_results/hydra_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    tracker = HydraExperimentTracker(output_dir)
    tracker.log_experiment_config(config)
    
    model = setup_model_from_config(config)
    
    game_configs = prepare_game_configs_from_hydra(config)
    
    for game_config in game_configs:
        game_type = game_config["game_type"]
        args = game_config["args"]
        
        logger.info(f"Running {game_type} with {config.model.name}")
        
        result = run_game((game_type, args, None, 0, output_dir))
        
        if result:
            result["model"] = config.model.name
            result["timestamp"] = datetime.now().isoformat()
            tracker.log_result(result)
            
    tracker.save_summary()
    
    logger.info(f"Benchmark completed. Results saved in {output_dir}")

if __name__ == "__main__":
    main() 