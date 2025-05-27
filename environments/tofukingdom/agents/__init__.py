from environments.tofukingdom.agents.role_agent import RoleAgent
from environments.tofukingdom.agents.game_controller import GameController

ROLES = {
    "Princess": "Princess",
    "Chef": "Chef",
    "Queen": "Queen",
    "Minister": "Minister",
    "Guard": "Guard",
    "Maid": "Maid",
    "Spy": "Spy"
}

TEAMS = {
    "Princess": "Princess",
    "Chef": "Princess",
    "Queen": "Queen",
    "Minister": "Queen",
    "Guard": "Queen",
    "Maid": "Neutral",
    "Spy": "Neutral"
}

PrincessAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Princess")
ChefAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Chef")
GuardAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Guard")
MaidAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Maid")
MinisterAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Minister")
QueenAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Queen")
SpyAgent = lambda llm, player_name, all_players: RoleAgent.create(llm, player_name, all_players, "Spy")