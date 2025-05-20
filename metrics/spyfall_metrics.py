from typing import Dict, List, Tuple, Union, Any, Set, Optional
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
import torch
import os
from datetime import datetime

from metrics.base_metrics import BaseMetrics

class SpyfallMetrics(BaseMetrics):
    """
    Metrics collection and computation for the Spyfall game.
    
    This class extends BaseMetrics to handle Spyfall-specific metrics:
    - Spy survival and detection rates
    - Description analysis (specificity, vagueness)
    - Voting pattern analysis
    - Various performance metrics
    
    Additionally includes all common metrics from BaseMetrics.
    """
    
    # Spyfall specific event types
    EVENT_DESCRIPTION = "description"
    EVENT_VOTE = "vote"
    EVENT_ROLE_ASSIGNED = "role_assigned"
    
    # Spyfall specific LLM evaluation templates
    SPYFALL_GAME_EVALUATION_TEMPLATE = """
    Evaluate this Spyfall game based on the provided information:
    
    Game setup:
    - Spy word: {spy_word}
    - Villager word: {villager_word}
    - Spy player: {spy_name}
    - Villagers: {villagers}
    - Winner: {winner}
    
    Game summary:
    {game_summary}
    
    Please provide a detailed analysis of:
    1. Spy strategy and effectiveness (score 1-10)
    2. Villagers coordination and detection skills (score 1-10)
    3. Quality of word descriptions by all players (score 1-10)
    4. Voting strategy effectiveness (score 1-10)
    5. Key moments that determined the game outcome
    
    For each player, provide an individual performance evaluation with score (1-10).
    
    Finally, suggest how the LLM agents could have played better in this game scenario.
    """
    
    SPYFALL_ROUND_EVALUATION_TEMPLATE = """
    Evaluate this round (Round {round_num}) of Spyfall:
    
    Current game state:
    - Living players: {living_players}
    - Spy: {spy_name} (this is hidden from players)
    - Game words: Spy word is "{spy_word}", Villager word is "{villager_word}"
    
    Round activities:
    {round_summary}
    
    Please analyze:
    1. How well did the spy blend in? (score 1-10)
    2. How effective were the villagers at identifying suspicious behavior? (score 1-10)
    3. How strategic were the voting decisions? (score 1-10)
    4. Which player had the most impact on this round? Why?
    
    Provide a brief evaluation of each player's performance this round with a score (1-10).
    """
    
    SPYFALL_DESCRIPTION_EVALUATION_TEMPLATE = """
    Evaluate this player's description in Spyfall:
    
    Player: {player_name} ({role})
    Description: "{description}"
    
    Word context:
    - Spy word: {spy_word}
    - Villager word: {villager_word}
    
    Please analyze:
    1. Appropriateness of the description (score 1-10)
    2. Strategy effectiveness (score 1-10)
    3. Balance between information sharing and self-protection (score 1-10)
    
    For a spy: How well did they blend in without revealing ignorance?
    For a villager: How well did they communicate the word without being too obvious?
    
    Provide a brief analysis explaining your scores and reasoning.
    """
    
    SPYFALL_VOTE_EVALUATION_TEMPLATE = """
    Evaluate this player's voting decision in Spyfall:
    
    Player: {player_name} ({role})
    Voted for: {vote_for}
    Reasoning: "{reasoning}"
    
    Current game state:
    - Living players: {living_players}
    - Round: {round_num}
    - Previous voting pattern: {voting_history}
    
    Please analyze:
    1. Strategic value of this vote (score 1-10)
    2. Reasoning quality (score 1-10)
    3. Impact on game outcome (score 1-10)
    
    Provide a brief analysis explaining your assessment of this voting decision.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the SpyfallMetrics collector.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Additional metadata for the game session
        """
        super().__init__("spyfall", metadata)
        self.descriptions = {}
        self.votes = {}
        self.spy_name = None
        self.spy_index = None
        self.villagers = []
        self.players = []
        self.spy_caught = False
        self.winner = None
        self.spy_word = None
        self.villager_word = None
        self.round_summaries = {}
        self.current_round = 0
    
    def add_game_words(self, spy_word: str, villager_word: str) -> None:
        """
        Add the game words to the metrics.
        
        Args:
            spy_word (str): The word given to the spy
            villager_word (str): The word given to the villagers
        """
        self.spy_word = spy_word
        self.villager_word = villager_word
        self.add_metadata("spy_word", spy_word)
        self.add_metadata("villager_word", villager_word)
    
    def evaluate_game(self) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of the entire game.
        
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        # Build game summary from events
        descriptions_events = [e for e in self.events if e["type"] == self.EVENT_DESCRIPTION]
        vote_events = [e for e in self.events if e["type"] == self.EVENT_VOTE]
        
        # Format descriptions
        descriptions_summary = []
        for event in descriptions_events:
            player = event["data"].get("player", "Unknown")
            description = event["data"].get("description", "No description")
            is_spy = event["data"].get("is_spy", False)
            role = "Spy" if is_spy else "Villager"
            descriptions_summary.append(f"{player} ({role}): \"{description}\"")
        
        # Format votes
        votes_summary = []
        for event in vote_events:
            voter = event["data"].get("voter", "Unknown")
            vote_for = event["data"].get("vote_for", "Unknown")
            is_spy_voter = event["data"].get("is_spy_voter", False)
            role = "Spy" if is_spy_voter else "Villager"
            votes_summary.append(f"{voter} ({role}) voted for {vote_for}")
        
        # Build game summary
        game_summary = "Descriptions:\n" + "\n".join(descriptions_summary) + "\n\n"
        game_summary += "Votes:\n" + "\n".join(votes_summary)
        
        # Context for evaluation
        context = {
            "spy_word": self.spy_word,
            "villager_word": self.villager_word,
            "spy_name": self.spy_name,
            "villagers": self.villagers,
            "winner": self.winner or "Unknown",
            "game_summary": game_summary
        }
        
        # Request evaluation using the Spyfall-specific template
        return self.record_llm_evaluation("game", context, self.SPYFALL_GAME_EVALUATION_TEMPLATE)
    
    def evaluate_round(self, round_num: int) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific round.
        
        Args:
            round_num (int): Round number to evaluate
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        # Get round-specific events
        round_events = []
        in_target_round = False
        
        for event in self.events:
            if event["type"] == self.EVENT_ROUND_START and event["data"].get("round_number") == round_num:
                in_target_round = True
                round_events.append(event)
                
            elif event["type"] == self.EVENT_ROUND_END and event["data"].get("round_number") == round_num:
                round_events.append(event)
                in_target_round = False
                
            elif in_target_round:
                round_events.append(event)
        
        # Extract descriptions and votes from round events
        descriptions = []
        votes = []
        
        for event in round_events:
            if event["type"] == self.EVENT_DESCRIPTION:
                player = event["data"].get("player", "Unknown")
                description = event["data"].get("description", "No description")
                is_spy = event["data"].get("is_spy", False)
                role = "Spy" if is_spy else "Villager"
                descriptions.append(f"{player} ({role}): \"{description}\"")
                
            elif event["type"] == self.EVENT_VOTE:
                voter = event["data"].get("voter", "Unknown")
                vote_for = event["data"].get("vote_for", "Unknown")
                is_spy_voter = event["data"].get("is_spy_voter", False)
                role = "Spy" if is_spy_voter else "Villager"
                votes.append(f"{voter} ({role}) voted for {vote_for}")
        
        # Build round summary
        round_summary = "Descriptions:\n" + "\n".join(descriptions) + "\n\n"
        round_summary += "Votes:\n" + "\n".join(votes)
        
        # Get living players for this round
        living_players = None
        for event in round_events:
            if event["type"] == self.EVENT_ROUND_START:
                living_players = event["data"].get("living_players", None)
                break
        
        # Context for evaluation
        context = {
            "round_num": round_num,
            "living_players": living_players or self.players,
            "spy_name": self.spy_name,
            "spy_word": self.spy_word,
            "villager_word": self.villager_word,
            "round_summary": round_summary
        }
        
        # Request evaluation using the Spyfall-specific template
        return self.record_llm_evaluation("round", context, self.SPYFALL_ROUND_EVALUATION_TEMPLATE)
    
    def evaluate_description(self, player_name: str, description: str, is_spy: bool) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a player's description.
        
        Args:
            player_name (str): Name of the player
            description (str): The description provided
            is_spy (bool): Whether the player is the spy
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        # Context for evaluation
        context = {
            "player_name": player_name,
            "role": "Spy" if is_spy else "Villager",
            "description": description,
            "spy_word": self.spy_word,
            "villager_word": self.villager_word
        }
        
        # Request evaluation using the Spyfall-specific template
        return self.record_llm_evaluation("description", context, self.SPYFALL_DESCRIPTION_EVALUATION_TEMPLATE)
    
    def evaluate_vote(self, voter: str, vote_for: str, is_spy_voter: bool, reasoning: str = "") -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a player's vote.
        
        Args:
            voter (str): Name of the voting player
            vote_for (str): Name of the player being voted for
            is_spy_voter (bool): Whether the voter is the spy
            reasoning (str, optional): Player's reasoning for the vote
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        # Get voting history for context
        voting_history = []
        for player, votes_list in self.votes.items():
            for vote in votes_list:
                if vote["round"] < self.current_round:  # Only include votes from previous rounds
                    voting_history.append(f"{player} voted for {vote['vote_for']} in round {vote['round']}")
        
        # Get current living players
        living_players = self.get_living_players()
        
        # Context for evaluation
        context = {
            "player_name": voter,
            "role": "Spy" if is_spy_voter else "Villager",
            "vote_for": vote_for,
            "reasoning": reasoning,
            "living_players": living_players,
            "round_num": self.current_round,
            "voting_history": voting_history
        }
        
        # Request evaluation using the Spyfall-specific template
        return self.record_llm_evaluation("vote", context, self.SPYFALL_VOTE_EVALUATION_TEMPLATE)
    
    def get_living_players(self) -> List[str]:
        """
        Get the list of living players from the most recent round start event.
        
        Returns:
            List[str]: List of living player names
        """
        round_start_events = [e for e in self.events if e["type"] == self.EVENT_ROUND_START]
        
        if round_start_events:
            latest_round = round_start_events[-1]
            return latest_round["data"].get("living_players", self.players)
        
        return self.players
    
    def record_description(self, player_name: str, description: str, is_spy: bool) -> None:
        """
        Record a player's description.
        
        Args:
            player_name (str): Name of the player
            description (str): Description provided by the player
            is_spy (bool): Whether the player is the spy
        """
        self.descriptions[player_name] = {
            "text": description,
            "is_spy": is_spy,
            "specificity": self.calculate_description_specificity(description),
            "perplexity": self.calculate_perplexity(description)
        }
        
        self.record_event(
            self.EVENT_DESCRIPTION,
            player=player_name,
            description=description,
            is_spy=is_spy
        )
        
        # If LLM evaluation is enabled, evaluate this description
        if self.use_llm_evaluation:
            evaluation = self.evaluate_description(player_name, description, is_spy)
            if evaluation:
                self.descriptions[player_name]["llm_evaluation"] = evaluation
    
    def record_vote(self, voter: str, vote_for: str, is_spy_voter: bool, reasoning: str = "") -> None:
        """
        Record a player's vote.
        
        Args:
            voter (str): Name of the player casting the vote
            vote_for (str): Name of the player being voted for
            is_spy_voter (bool): Whether the voting player is the spy
            reasoning (str, optional): Player's reasoning for the vote
        """
        if voter not in self.votes:
            self.votes[voter] = []
            
        vote_data = {
            "vote_for": vote_for,
            "is_spy_voter": is_spy_voter,
            "voted_for_spy": vote_for == self.spy_name,
            "round": self.get_current_round(),
            "reasoning": reasoning
        }
        
        self.votes[voter].append(vote_data)
        
        self.record_event(
            self.EVENT_VOTE,
            voter=voter,
            vote_for=vote_for,
            is_spy_voter=is_spy_voter,
            voted_for_spy=vote_for == self.spy_name,
            round=self.get_current_round(),
            reasoning=reasoning
        )
        
        # If LLM evaluation is enabled, evaluate this vote
        if self.use_llm_evaluation:
            evaluation = self.evaluate_vote(voter, vote_for, is_spy_voter, reasoning)
            if evaluation:
                self.votes[voter][-1]["llm_evaluation"] = evaluation
    
    def record_role_assignment(self, players: List[str], spy_index: int, spy_name: str) -> None:
        """
        Record the role assignments at the start of the game.
        
        Args:
            players (List[str]): List of all player names
            spy_index (int): Index of the spy in the players list
            spy_name (str): Name of the spy
        """
        self.players = players
        self.spy_index = spy_index
        self.spy_name = spy_name
        self.villagers = [p for p in players if p != spy_name]
        
        self.add_metadata("players", players)
        self.add_metadata("spy_index", spy_index)
        self.add_metadata("spy_name", spy_name)
        
        for i, player in enumerate(players):
            is_spy = (i == spy_index - 1)  # Adjust for 1-indexing if needed
            
            self.record_event(
                self.EVENT_ROLE_ASSIGNED,
                player=player,
                is_spy=is_spy,
                role="spy" if is_spy else "villager"
            )
    
    def record_game_end(self, winner: str, spy_caught: bool) -> None:
        """
        Record the game end result.
        
        Args:
            winner (str): Who won the game ("spy" or "villager")
            spy_caught (bool): Whether the spy was caught
        """
        self.winner = winner
        self.spy_caught = spy_caught
        
        self.add_metadata("winner", winner)
        self.add_metadata("spy_caught", spy_caught)
        
        self.record_event(
            self.EVENT_GAME_END,
            winner=winner,
            spy_caught=spy_caught
        )
    
    def get_current_round(self) -> int:
        """
        Get the current game round based on recorded events.
        
        Returns:
            int: Current round number
        """
        round_start_events = [e for e in self.events if e["type"] == self.EVENT_ROUND_START]
        return len(round_start_events)
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all Spyfall metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        # Call the parent method to get common metrics
        base_metrics = super().compute_all()
        
        # Add Spyfall-specific metrics
        spyfall_metrics = {
            "description_metrics": self._compute_description_metrics(),
            "voting_metrics": self._compute_voting_metrics(),
            "game_outcome": {
                "winner": self.winner,
                "spy_caught": self.spy_caught,
                "rounds_played": self.get_current_round()
            }
        }
        
        # Merge metrics
        self.computed_metrics.update(spyfall_metrics)
        
        return self.computed_metrics
    
    def _compute_description_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to descriptions.
        
        Returns:
            Dict[str, Any]: Description metrics
        """
        # Basic metrics
        description_metrics = {
            "by_player": self.descriptions,
            "avg_specificity": sum(d["specificity"] for d in self.descriptions.values()) / len(self.descriptions) if self.descriptions else 0,
            "avg_perplexity": sum(d["perplexity"] for d in self.descriptions.values()) / len(self.descriptions) if self.descriptions else 0
        }
        
        # Spy vs. Villager comparison
        spy_descriptions = [d for p, d in self.descriptions.items() if d["is_spy"]]
        villager_descriptions = [d for p, d in self.descriptions.items() if not d["is_spy"]]
        
        if spy_descriptions:
            description_metrics["spy_avg_specificity"] = sum(d["specificity"] for d in spy_descriptions) / len(spy_descriptions)
            description_metrics["spy_avg_perplexity"] = sum(d["perplexity"] for d in spy_descriptions) / len(spy_descriptions)
        
        if villager_descriptions:
            description_metrics["villager_avg_specificity"] = sum(d["specificity"] for d in villager_descriptions) / len(villager_descriptions)
            description_metrics["villager_avg_perplexity"] = sum(d["perplexity"] for d in villager_descriptions) / len(villager_descriptions)
        
        # Vagueness scores
        if self.descriptions:
            description_metrics["vagueness_scores"] = self.vagueness_score([d["text"] for d in self.descriptions.values()])
        
        return description_metrics
    
    def _compute_voting_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to voting patterns.
        
        Returns:
            Dict[str, Any]: Voting metrics
        """
        voting_metrics = {
            "by_player": self.votes,
            "votes_for_spy": 0,
            "spy_votes_accuracy": 0,
            "villager_votes_accuracy": 0
        }
        
        # Count votes for spy
        spy_votes = 0
        total_votes = 0
        
        # Spy voting accuracy
        spy_correct_votes = 0
        spy_total_votes = 0
        
        # Villager voting accuracy
        villager_correct_votes = 0
        villager_total_votes = 0
        
        for player, vote_list in self.votes.items():
            for vote in vote_list:
                total_votes += 1
                if vote["voted_for_spy"]:
                    spy_votes += 1
                    
                # Check if voter is spy and tracking their accuracy
                if vote["is_spy_voter"]:
                    spy_total_votes += 1
                    if not vote["voted_for_spy"]:  # Spy correctly voting for villager
                        spy_correct_votes += 1
                else:  # Voter is villager
                    villager_total_votes += 1
                    if vote["voted_for_spy"]:  # Villager correctly voting for spy
                        villager_correct_votes += 1
        
        if total_votes > 0:
            voting_metrics["votes_for_spy"] = spy_votes / total_votes
            
        if spy_total_votes > 0:
            voting_metrics["spy_votes_accuracy"] = spy_correct_votes / spy_total_votes
            
        if villager_total_votes > 0:
            voting_metrics["villager_votes_accuracy"] = villager_correct_votes / villager_total_votes
        
        # Calculate vote influence metrics if needed
        voting_metrics["vote_influence"] = self.vote_influence_index()
        
        return voting_metrics
    
    # Methods ported from the original SpyfallMetrics class
    
    @staticmethod
    def calculate_description_specificity(description: str, abstract_words: Set[str] = None) -> float:
        """
        Calculate how specific a description is by measuring the absence of abstract/vague terms.
        
        Args:
            description: The text description to analyze
            abstract_words: Set of abstract/vague words to check against
            
        Returns:
            Specificity score between 0 and 1 (higher is more specific)
        """
        if not description:
            return 0.0
            
        # Default set of vague/abstract words if none provided
        if abstract_words is None:
            abstract_words = {
                "thing", "something", "stuff", "item", "object", "entity", "device",
                "gadget", "material", "substance", "matter", "product", "element",
                "tool", "instrument", "implement", "apparatus", "equipment", "mechanism",
                "can", "could", "may", "might", "would", "should", "must", "will", 
                "general", "common", "usual", "regular", "normal", "ordinary", "typical",
                "standard", "basic", "fundamental", "conventional", "traditional",
                "it", "this", "that", "these", "those", "they", "them", "their",
                "often", "sometimes", "occasionally", "frequently", "generally", "usually",
                "possibly", "potentially", "perhaps", "maybe", "probably",
                "various", "different", "diverse", "several", "many", "few", "some"
            }
            
        # Clean and tokenize the description
        words = re.findall(r'\b\w+\b', description.lower())
        
        if not words:
            return 0.0
            
        # Count abstract words in the description
        abstract_count = sum(1 for word in words if word in abstract_words)
        
        # Calculate specificity score (1 - proportion of abstract words)
        specificity = 1.0 - (abstract_count / len(words))
        
        return specificity
    
    @staticmethod
    def calculate_perplexity(text: str, api_key: str = None, model_name: str = "gpt-2") -> float:
        """
        Calculate the perplexity of text using a language model.
        
        Lower perplexity = more natural/coherent text
        Higher perplexity = more confusing/unusual text
        
        Args:
            text: Text to analyze
            api_key: API key for external model (if needed)
            model_name: Language model to use
            
        Returns:
            Perplexity score (lower is more coherent)
        """
        # Simple implementation for now - could be replaced with a real perplexity calculation
        # with an actual language model if needed
        if not text:
            return 0.0
            
        # This is a simplified perplexity estimation based on some text features
        # that might correlate with actual perplexity
        
        # Sentence length (longer sentences can be more complex)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(re.findall(r'\b\w+\b', s)) for s in sentences) / len(sentences) if sentences else 0
        
        # Vocabulary diversity
        words = re.findall(r'\b\w+\b', text.lower())
        vocab_diversity = len(set(words)) / len(words) if words else 0
        
        # Presence of unusual punctuation or patterns
        unusual_patterns = len(re.findall(r'[^a-zA-Z0-9 .!?,;:\'"-]', text))
        
        # Combine factors to estimate perplexity (completely heuristic)
        perplexity = (0.5 * avg_sentence_length) + (20 * vocab_diversity) + (2 * unusual_patterns)
        
        # Normalize to a reasonable range (0-20, with 10 being average)
        return min(20, max(0, perplexity))
    
    @staticmethod
    def vagueness_score(descriptions: List[str]) -> Dict[str, float]:
        """
        Calculate vagueness scores for a set of descriptions.
        
        Uses TF-IDF to identify content-rich words vs general words.
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            Dictionary with various vagueness metrics
        """
        if not descriptions or len(descriptions) < 2:
            return {"avg_vagueness": 0, "max_vagueness": 0, "min_vagueness": 0}
            
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.9
        )
        
        try:
            # Transform descriptions to TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            
            # Calculate average TF-IDF score for each description
            # Lower score = more vague (fewer distinctive words)
            # Higher score = more specific (more distinctive words)
            avg_tfidf_scores = tfidf_matrix.mean(axis=1).A1
            
            # Convert to vagueness score (1 - tfidf, so higher = more vague)
            vagueness_scores = 1 - (avg_tfidf_scores / avg_tfidf_scores.max())
            
            return {
                "avg_vagueness": float(np.mean(vagueness_scores)),
                "max_vagueness": float(np.max(vagueness_scores)),
                "min_vagueness": float(np.min(vagueness_scores)),
                "std_vagueness": float(np.std(vagueness_scores))
            }
        except:
            return {"avg_vagueness": 0, "max_vagueness": 0, "min_vagueness": 0, "std_vagueness": 0}
    
    def vote_influence_index(self) -> Dict[str, float]:
        """
        Calculate how influential each player's votes were.
        
        Returns:
            Dictionary mapping player names to influence scores
        """
        influence_scores = {}
        
        # If we don't have votes, return empty dict
        if not self.votes:
            return influence_scores
            
        # Count how many players voted for each target
        vote_counts = {}
        for player, vote_list in self.votes.items():
            for vote in vote_list:
                target = vote["vote_for"]
                if target not in vote_counts:
                    vote_counts[target] = 0
                vote_counts[target] += 1
        
        # Define influence as voting with the majority
        for player, vote_list in self.votes.items():
            majority_votes = 0
            total_votes = len(vote_list)
            
            for vote in vote_list:
                target = vote["vote_for"]
                if target in vote_counts and vote_counts[target] > 1:  # If voted for someone others also voted for
                    majority_votes += 1
            
            # Influence is the proportion of votes that aligned with others
            influence_scores[player] = majority_votes / total_votes if total_votes > 0 else 0
        
        return influence_scores 