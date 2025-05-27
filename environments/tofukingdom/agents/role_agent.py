from environments.tofukingdom.agents.base_agent import TofuKingdomAgent
from typing import Dict, List, Any, Optional, Tuple, Literal

RoleType = Literal["Prince", "Princess", "Queen", "Minister", "Chef", "Guard", "Maid", "Spy"]
TruthBehavior = Literal["truth", "lie", "choice"]

class RoleAgent(TofuKingdomAgent):
    """
    Unified role agent for the TofuKingdom game.
    
    This agent can represent any role in the game (except Prince) and handles
    truth/lying behavior according to the role's rules.
    
    Args:
        llm: LangChain-compatible language model
        player_name: Name of the player
        all_players: List of all players in the game
        role: Role of the agent (Princess, Queen, etc.)
    """
    ROLE_BEHAVIORS: Dict[RoleType, TruthBehavior] = {
        "Princess": "truth",
        "Chef": "truth",
        "Queen": "lie",
        "Minister": "lie",
        "Guard": "lie",
        "Maid": "choice",
        "Spy": "choice"
    }
    
    ROLE_TEAMS: Dict[RoleType, str] = {
        "Princess": "Princess",
        "Chef": "Princess",
        "Queen": "Queen",
        "Minister": "Queen",
        "Guard": "Queen",
        "Maid": "Neutral",
        "Spy": "Neutral"
    }
    
    def __init__(self, llm, player_name: str, all_players: List[str], role: RoleType) -> None:
        super().__init__(llm, player_name, all_players, role)
        self.truth_behavior = self.ROLE_BEHAVIORS.get(role, "choice")
        self.team = self.ROLE_TEAMS.get(role, "Neutral")
    
    @classmethod
    def create(cls, llm, player_name: str, all_players: List[str], role: RoleType) -> "RoleAgent":
        """
        Factory method to create a role agent.
        
        Args:
            llm: LangChain-compatible language model
            player_name: Name of the player
            all_players: List of all players in the game
            role: Role of the agent (Princess, Queen, etc.)
            
        Returns:
            A RoleAgent instance configured for the specified role
        """
        return cls(llm, player_name, all_players, role)
        
    def answer_question(self, question: str, identities: Dict[str, str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process a question from the Prince and return an answer based on the role's truth behavior.
        
        Args:
            question: The question asked by the Prince
            identities: Dictionary mapping player names to their roles
            
        Returns:
            Tuple of answer text and complete response with thought
        """
        self.private_history.append({"role": "user", "content": f"Prince asks: {question}"})
        
        return self.process_question(identities)
    
    def get_truth_behavior(self) -> TruthBehavior:
        """
        Returns the truth behavior for this role.
        
        Returns:
            String indicating whether the role must tell the truth, lie, or has a choice
        """
        return self.truth_behavior 