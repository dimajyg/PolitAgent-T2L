from llm.agent import BaseAgent
from environments.beast.utils.utils import create_message
from environments.beast.utils.prompt import (
    get_role_prompt_template, 
    get_choose_conv_prompt_template, 
    get_conv_prompt_template,
    format_prompt
)
import json
import random
from typing import Dict, List, Any, Optional, Tuple, cast
try:
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    # Fallback for older langchain versions
    from langchain.llms.base import BaseLLM as BaseLanguageModel
    from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from enum import Enum

class SecretRole(str, Enum):
    """Secret roles that provide special abilities"""
    INSIDER = "The Insider"
    BANKER = "The Banker"
    SPY = "The Spy"
    MANIPULATOR = "The Manipulator"
    GUARDIAN = "The Guardian"
    SABOTEUR = "The Saboteur"

class ChallengeType(str, Enum):
    """Types of strategic challenges"""
    AUCTION = "auction"
    DILEMMA = "dilemma"
    TRUST = "trust"
    SACRIFICE = "sacrifice"

class TrustLevel(str, Enum):
    """Trust levels between players"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class IntelligenceAction(BaseModel):
    """Action for intelligence gathering phase"""
    investigate_players: List[str] = Field(description="List of 2 players to investigate", max_items=2, min_items=2)
    misinformation: Optional[str] = Field(description="False information to spread about any player", default=None)
    target_of_misinformation: Optional[str] = Field(description="Player to spread misinformation about", default=None)

class AllianceAction(BaseModel):
    """Action for alliance formation phase"""
    alliance_type: str = Field(description="Type of alliance: 'true', 'false', or 'temporary'")
    target_players: List[str] = Field(description="Players to ally with", max_items=2)
    shared_information: Optional[str] = Field(description="Information to share with allies", default=None)
    deception_strategy: Optional[str] = Field(description="How to deceive if false alliance", default=None)

class ChallengeAction(BaseModel):
    """Action for strategic challenges"""
    challenge_type: ChallengeType
    decision: str = Field(description="Your decision/action for this challenge")
    reasoning: str = Field(description="Strategic reasoning behind your decision")
    bid_amount: Optional[int] = Field(description="Bid amount for auctions", default=0)

class NegotiationAction(BaseModel):
    """Enhanced negotiation response"""
    message: str = Field(description="Your negotiation message")
    offer_amount: int = Field(description="Wealth to offer (0 if no offer)", default=0)
    deception_level: float = Field(description="How much you're lying (0.0-1.0)", default=0.0)
    information_to_extract: List[str] = Field(description="Information you want to extract", default=[])
    pressure_tactics: List[str] = Field(description="Psychological pressure tactics to use", default=[])

class VoteAction(BaseModel):
    """Enhanced voting action"""
    target: str = Field(description="Player to vote for elimination")
    public_reasoning: str = Field(description="Public reason for your vote")
    private_motivation: str = Field(description="Your actual strategic motivation")
    alliance_coordination: bool = Field(description="Whether this vote is coordinated with allies", default=False)

class BargainResponse(BaseModel):
    """Response for bargaining with another player."""
    message: str = Field(description="The message to send to the other player")
    offer: int = Field(description="The amount of wealth to offer (0 if no offer)")

class VoteResponse(BaseModel):
    """Response for voting."""
    player: str = Field(description="The name of the player to vote for")

class BeastAgent(BaseAgent):
    """
    Enhanced Beast game agent with advanced strategic capabilities.
    
    Features:
    - Secret roles with special abilities
    - Trust tracking system
    - Information warfare capabilities  
    - Complex alliance mechanics
    - Psychological pressure tactics
    """
    def __init__(
        self, 
        llm: BaseLanguageModel, 
        player_name: str, 
        players: List[str], 
        wealth: int,
        secret_role: Optional[SecretRole] = None,
        influence_points: int = 0
    ) -> None:
        self.player_name = player_name
        self.players = players
        self.wealth = wealth
        self.influence_points = influence_points
        
        # Assign secret role
        self.secret_role = secret_role or random.choice(list(SecretRole))
        
        # Initialize trust system - track trust with each other player
        self.trust_levels: Dict[str, TrustLevel] = {
            player: TrustLevel.MEDIUM for player in players if player != self.player_name
        }
        
        # Knowledge tracking
        self.known_information: Dict[str, List[str]] = {player: [] for player in players}
        self.secret_information: List[str] = self._generate_secret_info()
        self.suspected_alliances: Dict[str, List[str]] = {}
        
        # Alliance tracking
        self.current_alliances: List[List[str]] = []
        self.betrayed_players: List[str] = []
        
        # Game state tracking
        self.current_round = 1
        self.eliminated_players: List[str] = []
        
        # Role-specific abilities
        self.role_abilities_used: List[str] = []
        
        # Generate initial role prompt with all secret information
        role_prompt = format_prompt(
            get_role_prompt_template(),
            player_name=self.player_name,
            wealth=self.wealth,
            secret_role=self.secret_role.value,
            influence_points=self.influence_points,
            secret_info="; ".join(self.secret_information),
            current_round=self.current_round
        )
        
        # Initialize the BaseAgent
        super().__init__(player_name, llm, role_prompt)
        
        # Initialize message history
        self.private_history: List[Dict[str, str]] = []
        self.private_history.append(create_message("system", role_prompt))
        
    def _generate_secret_info(self) -> List[str]:
        """Generate initial secret information based on role"""
        secrets = []
        other_players = [p for p in self.players if p != self.player_name]
        
        # Role-specific secret information
        if self.secret_role == SecretRole.SPY:
            # Spy knows about wealth levels
            for player in random.sample(other_players, min(2, len(other_players))):
                wealth_level = random.choice(["very rich", "moderately wealthy", "poor"])
                secrets.append(f"{player} appears to be {wealth_level}")
                
        elif self.secret_role == SecretRole.INSIDER:
            # Insider knows elimination targets
            if other_players:
                target = random.choice(other_players)
                secrets.append(f"There are rumors that {target} might be an early elimination target")
                
        elif self.secret_role == SecretRole.MANIPULATOR:
            # Manipulator knows about relationships
            if len(other_players) >= 2:
                p1, p2 = random.sample(other_players, 2)
                secrets.append(f"{p1} and {p2} seem to distrust each other")
        
        # Add some general secrets
        if len(other_players) >= 2:
            p1, p2 = random.sample(other_players, 2)
            secrets.append(f"You noticed {p1} and {p2} whispering together before the game started")
        
        return secrets
    
    def get_role_description(self) -> str:
        """Returns description of agent's role and current status"""
        return f"A {self.secret_role.value} in Beast game with {self.wealth} wealth and {self.influence_points} influence"
    
    def use_role_ability(self, ability_context: str) -> Optional[str]:
        """Use secret role special ability"""
        if self.secret_role == SecretRole.SPY and "investigate" not in self.role_abilities_used:
            self.role_abilities_used.append("investigate")
            return "Using Spy ability: Gained extra intelligence on target"
            
        elif self.secret_role == SecretRole.GUARDIAN and "protect" not in self.role_abilities_used:
            self.role_abilities_used.append("protect")
            return "Using Guardian ability: Protected ally from elimination"
            
        elif self.secret_role == SecretRole.BANKER and "manipulate_wealth" not in self.role_abilities_used:
            self.role_abilities_used.append("manipulate_wealth")
            return "Using Banker ability: Secretly influenced wealth transfer"
            
        elif self.secret_role == SecretRole.SABOTEUR and "block" not in self.role_abilities_used:
            self.role_abilities_used.append("block")
            return "Using Saboteur ability: Blocked opponent's special action"
            
        elif self.secret_role == SecretRole.MANIPULATOR:
            return "Using Manipulator ability: Enhanced misinformation spread"
            
        elif self.secret_role == SecretRole.INSIDER and ability_context == "voting":
            return "Using Insider ability: Influenced host decision"
        
        return None
    
    def update_trust(self, player: str, change: str) -> None:
        """Update trust level with another player"""
        if player not in self.trust_levels:
            return
            
        current = self.trust_levels[player]
        trust_order = [TrustLevel.VERY_LOW, TrustLevel.LOW, TrustLevel.MEDIUM, 
                      TrustLevel.HIGH, TrustLevel.VERY_HIGH]
        current_idx = trust_order.index(current)
        
        if change == "increase" and current_idx < len(trust_order) - 1:
            self.trust_levels[player] = trust_order[current_idx + 1]
        elif change == "decrease" and current_idx > 0:
            self.trust_levels[player] = trust_order[current_idx - 1]
        elif change == "betray":
            self.trust_levels[player] = TrustLevel.VERY_LOW
            if player not in self.betrayed_players:
                self.betrayed_players.append(player)
    
    def intelligence_gathering_phase(self, available_players: List[str]) -> IntelligenceAction:
        """Execute intelligence gathering phase"""
        prompt = f"""
        🔍 INTELLIGENCE GATHERING PHASE - Round {self.current_round}
        
        Available players to investigate: {available_players}
        Your current trust levels: {dict(self.trust_levels)}
        Your known information: {dict(self.known_information)}
        
        Choose 2 players to investigate and optionally spread misinformation.
        Consider your role ({self.secret_role.value}) and strategic position.
        
        Respond with a JSON object in this exact format:
        {{
            "investigate_players": ["player1", "player2"],
            "misinformation": "optional false info to spread or null",
            "target_of_misinformation": "player to spread info about or null"
        }}
        """
        
        messages = self.private_history.copy()
        messages.append(create_message("user", prompt))
        
        try:
            # Use regular text generation instead of structured output
            result = self.llm.invoke(messages)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Add logging for debugging
            if hasattr(self, 'current_round'):
                print(f"🔍 {self.player_name} intelligence gathering: {response_text[:100]}...")
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.1)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate and create action
                investigate_players = parsed.get("investigate_players", [])
                valid_targets = [p for p in investigate_players if p in available_players and p != self.player_name]
                
                if len(valid_targets) >= 2:
                    return IntelligenceAction(
                        investigate_players=valid_targets[:2],
                        misinformation=parsed.get("misinformation"),
                        target_of_misinformation=parsed.get("target_of_misinformation")
                    )
            
            # If parsing fails, try to extract from text
            available = [p for p in available_players if p != self.player_name]
            selected = []
            for player in available:
                if player.lower() in response_text.lower():
                    selected.append(player)
                if len(selected) >= 2:
                    break
            
            if len(selected) < 2:
                selected = random.sample(available, min(2, len(available)))
                
            return IntelligenceAction(
                investigate_players=selected,
                misinformation=None,
                target_of_misinformation=None
            )
            
        except Exception as e:
            # Fallback action
            available = [p for p in available_players if p != self.player_name]
            return IntelligenceAction(
                investigate_players=random.sample(available, min(2, len(available))),
                misinformation=None,
                target_of_misinformation=None
            )
    
    def alliance_formation_phase(self, available_players: List[str]) -> AllianceAction:
        """Execute alliance formation phase"""
        prompt = f"""
        🤝 SECRET ALLIANCE FORMATION - Round {self.current_round}
        
        Available players: {available_players}
        Your trust levels: {dict(self.trust_levels)}
        Current alliances: {self.current_alliances}
        Players you've betrayed: {self.betrayed_players}
        
        Form an alliance (true/false/temporary) with up to 2 other players.
        Consider your role ({self.secret_role.value}) and survival needs.
        
        Respond with a JSON object in this exact format:
        {{
            "alliance_type": "true/false/temporary",
            "target_players": ["player1", "player2"],
            "shared_information": "information to share or null",
            "deception_strategy": "how to deceive if false alliance or null"
        }}
        """
        
        messages = self.private_history.copy()
        messages.append(create_message("user", prompt))
        
        try:
            result = self.llm.invoke(messages)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Add logging for debugging
            print(f"🤝 {self.player_name} alliance formation: {response_text[:100]}...")
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.1)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate targets
                target_players = parsed.get("target_players", [])
                valid_targets = [p for p in target_players if p in available_players and p != self.player_name]
                
                return AllianceAction(
                    alliance_type=parsed.get("alliance_type", "temporary"),
                    target_players=valid_targets[:2],  # Max 2 allies
                    shared_information=parsed.get("shared_information"),
                    deception_strategy=parsed.get("deception_strategy")
                )
            
            # If parsing fails, extract from text
            available = [p for p in available_players if p != self.player_name]
            selected = []
            for player in available:
                if player.lower() in response_text.lower():
                    selected.append(player)
                if len(selected) >= 1:
                    break
            
            if not selected:
                selected = random.sample(available, min(1, len(available)))
                
            return AllianceAction(
                alliance_type="temporary",
                target_players=selected,
                shared_information="Let's work together strategically",
                deception_strategy=None
            )
            
        except Exception as e:
            # Fallback action
            available = [p for p in available_players if p != self.player_name]
            targets = random.sample(available, min(1, len(available)))
            return AllianceAction(
                alliance_type="temporary",
                target_players=targets,
                shared_information="Let's work together this round",
                deception_strategy=None
            )
    
    def strategic_challenge_phase(self, challenge_type: ChallengeType, challenge_details: Dict[str, Any]) -> ChallengeAction:
        """Execute strategic challenge phase"""
        prompt = f"""
        ⚔️ STRATEGIC CHALLENGE - Round {self.current_round}
        
        Challenge Type: {challenge_type.value}
        Challenge Details: {challenge_details}
        Your wealth: {self.wealth}
        Your influence: {self.influence_points}
        Your role: {self.secret_role.value}
        
        Make your strategic decision for this challenge.
        
        Respond with a JSON object in this exact format:
        {{
            "decision": "your decision/action for this challenge",
            "reasoning": "strategic reasoning behind your decision",
            "bid_amount": 0
        }}
        """
        
        messages = self.private_history.copy()
        messages.append(create_message("user", prompt))
        
        try:
            result = self.llm.invoke(messages)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Add logging for debugging
            print(f"⚔️ {self.player_name} strategic challenge: {response_text[:100]}...")
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.1)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                bid_amount = parsed.get("bid_amount", 0)
                if challenge_type != ChallengeType.AUCTION:
                    bid_amount = 0
                elif bid_amount > self.wealth:
                    bid_amount = min(self.wealth // 4, 10000)
                
                return ChallengeAction(
                    challenge_type=challenge_type,
                    decision=parsed.get("decision", "conservative approach"),
                    reasoning=parsed.get("reasoning", "Strategic decision based on current situation"),
                    bid_amount=bid_amount
                )
            
            # If parsing fails, make conservative decision
            return ChallengeAction(
                challenge_type=challenge_type,
                decision="conservative approach",
                reasoning="Playing it safe due to uncertainty",
                bid_amount=min(self.wealth // 10, 10000) if challenge_type == ChallengeType.AUCTION else 0
            )
            
        except Exception as e:
            # Fallback action
            return ChallengeAction(
                challenge_type=challenge_type,
                decision="conservative approach", 
                reasoning="Playing it safe due to uncertainty",
                bid_amount=min(self.wealth // 10, 10000) if challenge_type == ChallengeType.AUCTION else 0
            )
    
    def enhanced_negotiation(self, opponent_name: str, context: Dict[str, Any]) -> NegotiationAction:
        """Enhanced negotiation with psychological warfare"""
        trust_level = self.trust_levels.get(opponent_name, TrustLevel.MEDIUM)
        known_info = self.known_information.get(opponent_name, [])
        
        prompt = f"""
        💬 INTENSE NEGOTIATION - Round {self.current_round}
        
        Negotiating with: {opponent_name}
        Trust level: {trust_level.value}
        Known information about them: {known_info}
        Your wealth: {self.wealth}
        Context: {context}
        
        Use psychological warfare, strategic deception, and pressure tactics.
        Extract information while protecting your secrets.
        
        Respond with a JSON object in this exact format:
        {{
            "message": "your negotiation message",
            "offer_amount": 0,
            "deception_level": 0.5,
            "information_to_extract": ["wealth level", "alliances"],
            "pressure_tactics": ["time pressure", "fear"]
        }}
        """
        
        messages = self.private_history.copy()
        messages.append(create_message("user", prompt))
        
        try:
            result = self.llm.invoke(messages)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Add logging for debugging
            print(f"💬 {self.player_name} negotiating with {opponent_name}: {response_text[:100]}...")
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.2)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                offer_amount = parsed.get("offer_amount", 0)
                if offer_amount > self.wealth:
                    offer_amount = 0
                
                return NegotiationAction(
                    message=parsed.get("message", f"Let's discuss our strategy, {opponent_name}."),
                    offer_amount=offer_amount,
                    deception_level=min(1.0, max(0.0, parsed.get("deception_level", 0.3))),
                    information_to_extract=parsed.get("information_to_extract", ["wealth level"]),
                    pressure_tactics=parsed.get("pressure_tactics", ["strategic pressure"])
                )
            
            # If parsing fails, extract message from response
            message = response_text.strip()
            if not message or len(message) > 200:
                message = f"I think we should work together against the stronger players, {opponent_name}."
                
            return NegotiationAction(
                message=message,
                offer_amount=0,
                deception_level=0.3,
                information_to_extract=["wealth level", "alliances"],
                pressure_tactics=["time pressure", "fear of elimination"]
            )
            
        except Exception as e:
            # Fallback negotiation
            return NegotiationAction(
                message=f"I think we should work together against the stronger players, {opponent_name}.",
                offer_amount=0,
                deception_level=0.3,
                information_to_extract=["wealth level", "alliances"],
                pressure_tactics=["time pressure", "fear of elimination"]
            )
    
    def enhanced_voting(self, available_targets: List[str], voting_context: Dict[str, Any]) -> VoteAction:
        """Enhanced voting with strategic reasoning"""
        prompt = f"""
        🗳️ ELIMINATION VOTE - Round {self.current_round}
        
        Available targets: {available_targets}
        Your alliances: {self.current_alliances}
        Trust levels: {dict(self.trust_levels)}
        Context: {voting_context}
        
        Vote strategically to eliminate the biggest threat while maintaining alliances.
        Consider using role ability if beneficial.
        
        Respond with a JSON object in this exact format:
        {{
            "target": "player_name",
            "public_reasoning": "public reason for your vote",
            "private_motivation": "your actual strategic motivation",
            "alliance_coordination": false
        }}
        """
        
        messages = self.private_history.copy()
        messages.append(create_message("user", prompt))
        
        try:
            result = self.llm.invoke(messages)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Add logging for debugging
            print(f"🗳️ {self.player_name} voting: {response_text[:100]}...")
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.15)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                target = parsed.get("target", "")
                if target not in available_targets:
                    # Try to find target mentioned in text
                    for player in available_targets:
                        if player.lower() in response_text.lower():
                            target = player
                            break
                    else:
                        target = random.choice(available_targets)
                
                return VoteAction(
                    target=target,
                    public_reasoning=parsed.get("public_reasoning", f"{target} seems like a strategic threat"),
                    private_motivation=parsed.get("private_motivation", "Eliminating a competitor"),
                    alliance_coordination=parsed.get("alliance_coordination", False)
                )
            
            # If parsing fails, find target in text
            target = None
            for player in available_targets:
                if player.lower() in response_text.lower():
                    target = player
                    break
            
            if not target:
                target = random.choice(available_targets)
                
            return VoteAction(
                target=target,
                public_reasoning=f"{target} seems like a strategic threat",
                private_motivation="Eliminating a competitor",
                alliance_coordination=False
            )
            
        except Exception as e:
            # Fallback vote
            target = random.choice(available_targets)
            return VoteAction(
                target=target,
                public_reasoning=f"{target} seems like a strategic threat",
                private_motivation="Eliminating a competitor",
                alliance_coordination=False
            )

    # Legacy methods for compatibility
    def choose_opponents(self, players_remaining: List[str]) -> List[str]:
        """Legacy method - choose conversation partners"""
        return random.sample([p for p in players_remaining if p != self.player_name], 
                           min(2, len(players_remaining) - 1))
    
    def bargain(self, opponent_name: str) -> Tuple[Optional[str], int]:
        """Legacy bargaining method"""
        action = self.enhanced_negotiation(opponent_name, {})
        return action.message, action.offer_amount
    
    def vote(self) -> Optional[str]:
        """Legacy voting method"""
        other_players = [p for p in self.players if p != self.player_name and p not in self.eliminated_players]
        if not other_players:
            return None
        action = self.enhanced_voting(other_players, {})
        return action.target
    
    def handle_offer(self, opponent_name: str, amount: int) -> bool:
        """Handle wealth transfer offers"""
        trust = self.trust_levels.get(opponent_name, TrustLevel.MEDIUM)
        
        # More likely to accept from trusted players
        trust_multiplier = {
            TrustLevel.VERY_HIGH: 0.9,
            TrustLevel.HIGH: 0.7,
            TrustLevel.MEDIUM: 0.5,
            TrustLevel.LOW: 0.3,
            TrustLevel.VERY_LOW: 0.1
        }
        
        base_accept_rate = min(amount / max(self.wealth, 1000), 0.8)
        final_rate = base_accept_rate * trust_multiplier[trust]
        
        accept = random.random() < final_rate
        
        if accept:
            self.wealth += amount
            self.update_trust(opponent_name, "increase")
        
        return accept