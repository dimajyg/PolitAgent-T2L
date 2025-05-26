"""
Metrics module for collecting and computing game-specific and common metrics
across all PolitAgent environments.

This module provides a unified interface for tracking performance metrics,
analyzing model behavior, and comparing different models and strategies.
"""

from metrics.base_metrics import BaseMetrics
from metrics.spyfall_metrics import SpyfallMetrics
from metrics.askguess_metrics import AskGuessMetrics
from metrics.tofukingdom_metrics import TofuKingdomMetrics
from metrics.beast_metrics import BeastMetrics
from metrics.diplomacy_metrics import DiplomacyMetrics

__all__ = [
    "BaseMetrics",
    "SpyfallMetrics",
    "AskGuessMetrics",
    "TofuKingdomMetrics",
    "BeastMetrics",
    "DiplomacyMetrics"
]

METRICS_MAP = {
    "spyfall": SpyfallMetrics,
    "askguess": AskGuessMetrics,
    "beast": BeastMetrics,
    "tofukingdom": TofuKingdomMetrics,
    "diplomacy": DiplomacyMetrics
}

def get_metrics(game_name, **kwargs):
    """
    Возвращает класс метрик для указанной игры.
    """
    if game_name not in METRICS_MAP:
        return BaseMetrics(**kwargs)
    return METRICS_MAP[game_name](**kwargs) 