from environments.tofukingdom.game import TofuKingdomGame as LegacyTofuKingdomGame

# Use legacy version as default for now
TofuKingdomGame = LegacyTofuKingdomGame

__all__ = [
    "TofuKingdomGame", 
    "LegacyTofuKingdomGame"
]
