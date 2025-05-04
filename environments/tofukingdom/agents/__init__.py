from environments.tofukingdom.agents.prince_agent import PrinceAgent
from environments.tofukingdom.agents.role_agent import RoleAgent
from environments.tofukingdom.agents.game_controller import GameController

# Define roles for easier access
ROLES = {
    "Princess": "Princess",
    "Chef": "Chef",
    "Queen": "Queen",
    "Minister": "Minister",
    "Guard": "Guard",
    "Maid": "Maid",
    "Spy": "Spy"
}

# Define teams for easier access
TEAMS = {
    "Princess": "Princess",
    "Chef": "Princess",
    "Queen": "Queen",
    "Minister": "Queen",
    "Guard": "Queen",
    "Maid": "Neutral",
    "Spy": "Neutral"
}

# For backwards compatibility with imports
# These classes are no longer used directly, but kept for compatibility
PrincessAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Princess")
ChefAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Chef")
GuardAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Guard")
MaidAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Maid")
MinisterAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Minister")
QueenAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Queen")
SpyAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Spy")