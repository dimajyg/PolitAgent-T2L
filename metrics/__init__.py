"""
Metrics module for PolitAgent environments.
Provides comprehensive metrics and evaluation tools for various game environments.
"""

from .base_metrics import BaseMetrics
from .diplomacy_metrics import DiplomacyMetrics
from .beast_metrics import BeastMetrics
from .spyfall_metrics import SpyfallMetrics
from .askguess_metrics import AskGuessMetrics
from .tofukingdom_metrics import TofuKingdomMetrics

__all__ = [
    'BaseMetrics',
    'DiplomacyMetrics', 
    'BeastMetrics',
    'SpyfallMetrics',
    'AskGuessMetrics',
    'TofuKingdomMetrics'
]

METRICS_MAP = {
    "beast": BeastMetrics,
    "diplomacy": DiplomacyMetrics,
    "spyfall": SpyfallMetrics,
    "askguess": AskGuessMetrics,
    "tofukingdom": TofuKingdomMetrics
}

def get_metrics(game_name, **kwargs):
    """
    Возвращает класс метрик для указанной игры.
    """
    if game_name not in METRICS_MAP:
        return BaseMetrics(**kwargs)
    return METRICS_MAP[game_name](**kwargs) 