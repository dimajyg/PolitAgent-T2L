from typing import Dict, List, Tuple, Union, Any, Set, Optional
import numpy as np
import re
import os
from datetime import datetime
import json

from metrics.base_metrics import BaseMetrics

class BeastMetrics(BaseMetrics):
    """
    Metrics collection and computation for the Beast strategic wealth game.
    
    This class extends BaseMetrics to handle Beast-specific metrics:
    - Wealth accumulation and distribution
    - Conversation and bargaining effectiveness
    - Voting patterns and strategy
    - Player elimination analysis
    - LLM evaluation of game strategies and outcomes
    
    Additionally includes all common metrics from BaseMetrics.
    """
    
    # Beast specific event types
    EVENT_WEALTH_CHANGE = "wealth_change"
    EVENT_CONVERSATION = "conversation"
    EVENT_BARGAIN = "bargain"
    EVENT_TRANSFER = "wealth_transfer"
    EVENT_VOTE = "vote"
    EVENT_ELIMINATION = "player_elimination"
    
    # Beast specific LLM evaluation templates
    BEAST_GAME_EVALUATION_TEMPLATE = """
    Evaluate this Beast strategic wealth game based on the provided information:
    
    Game setup:
    - Initial wealth: {initial_wealth}
    - Final wealth: {final_wealth}
    
    Game summary:
    {game_summary}
    
    Please provide a detailed analysis of:
    1. Wealth accumulation strategies (score 1-10)
    2. Bargaining effectiveness for each player (score 1-10)
    3. Voting strategies and outcomes (score 1-10)
    4. Critical moments that determined player success/failure
    5. Overall strategic depth and fairness (score 1-10)
    
    For each player, provide an individual performance evaluation with score (1-10).
    
    Finally, suggest how the LLM agents could have played better in this game scenario.
    """
    
    BEAST_CONVERSATION_EVALUATION_TEMPLATE = """
    Evaluate this conversation in the Beast strategic wealth game:
    
    Players involved: {players}
    Player wealth before conversation: {wealth_before}
    
    Conversation transcript:
    {transcript}
    
    Wealth transfer outcome: {transfer_outcome}
    
    Please analyze:
    1. Persuasiveness and negotiation strategy (score 1-10)
    2. Strategic alignment with game objectives (score 1-10)
    3. Effectiveness in wealth accumulation/protection (score 1-10)
    
    Provide a brief analysis explaining your assessment of this conversation.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the BeastMetrics collector.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Additional metadata for the game session
        """
        super().__init__("beast", metadata)
        self.player_wealth = {}  # Track wealth over time
        self.initial_wealth = {}  # Initial wealth of each player
        self.conversations = []  # All conversations
        self.wealth_transfers = []  # All wealth transfers between players
        self.votes = []  # All votes cast
        self.eliminations = []  # Track eliminated players
        self.game_summary = []  # Summary of game events
        
    def record_initial_wealth(self, player_name: str, wealth: int) -> None:
        """
        Record a player's initial wealth.
        
        Args:
            player_name (str): Name of the player
            wealth (int): Initial wealth value
        """
        self.initial_wealth[player_name] = wealth
        if player_name not in self.player_wealth:
            self.player_wealth[player_name] = []
        
        self.player_wealth[player_name].append({
            "round": 0,
            "wealth": wealth,
            "timestamp": datetime.now().isoformat()
        })
        
        self.record_event(
            self.EVENT_WEALTH_CHANGE,
            player=player_name,
            wealth=wealth,
            round=0,
            change=0,
            reason="initial"
        )
        
        self.add_metadata("initial_wealth", self.initial_wealth)
    
    def update_player_wealth(self, player_name: str, wealth: int, 
                           round_num: int, reason: str = None) -> None:
        """
        Update a player's wealth at a specific round.
        
        Args:
            player_name (str): Name of the player
            wealth (int): New wealth value
            round_num (int): Round number
            reason (str, optional): Reason for wealth change
        """
        if player_name not in self.player_wealth:
            # Initialize if this is first record
            self.player_wealth[player_name] = []
            prev_wealth = 0
        else:
            # Get previous wealth
            prev_wealth = self.player_wealth[player_name][-1]["wealth"] if self.player_wealth[player_name] else 0
        
        # Calculate change
        change = wealth - prev_wealth
        
        # Add new wealth record
        self.player_wealth[player_name].append({
            "round": round_num,
            "wealth": wealth,
            "change": change,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        self.record_event(
            self.EVENT_WEALTH_CHANGE,
            player=player_name,
            wealth=wealth,
            round=round_num,
            change=change,
            reason=reason
        )
        
        # Add to game summary
        if change != 0:
            change_text = f"increased by {change}" if change > 0 else f"decreased by {abs(change)}"
            self.game_summary.append(f"Round {round_num}: {player_name}'s wealth {change_text} to {wealth} ({reason})")
    
    def record_conversation(self, player1: str, player2: str, messages: List[Dict[str, Any]], 
                           round_num: int, transfer_outcome: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a conversation between two players.
        
        Args:
            player1 (str): First player in conversation
            player2 (str): Second player in conversation
            messages (List[Dict[str, Any]]): List of messages exchanged 
            round_num (int): Round number
            transfer_outcome (Optional[Dict[str, Any]]): Wealth transfer outcome if any
        """
        conversation = {
            "players": [player1, player2],
            "messages": messages,
            "round": round_num,
            "transfer_outcome": transfer_outcome,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations.append(conversation)
        
        self.record_event(
            self.EVENT_CONVERSATION,
            players=[player1, player2],
            messages=messages,
            round=round_num,
            transfer_outcome=transfer_outcome
        )
        
        # Add to game summary
        message_count = len(messages)
        transfer_text = ""
        if transfer_outcome:
            from_player = transfer_outcome.get("from")
            to_player = transfer_outcome.get("to")
            amount = transfer_outcome.get("amount")
            transfer_text = f" resulting in {from_player} transferring {amount} to {to_player}"
        
        self.game_summary.append(f"Round {round_num}: {player1} and {player2} exchanged {message_count} messages{transfer_text}")
        
        # If LLM evaluation is enabled, evaluate this conversation
        if self.use_llm_evaluation and transfer_outcome:
            # Only evaluate conversations with transfers for efficiency
            evaluation = self.evaluate_conversation(player1, player2, messages, round_num, transfer_outcome)
            if evaluation:
                conversation["llm_evaluation"] = evaluation
    
    def record_wealth_transfer(self, from_player: str, to_player: str, amount: int, 
                              round_num: int, reason: str = "bargain") -> None:
        """
        Record a wealth transfer between players.
        
        Args:
            from_player (str): Player sending wealth
            to_player (str): Player receiving wealth
            amount (int): Amount transferred
            round_num (int): Round number
            reason (str): Reason for transfer (bargain, bonus, etc.)
        """
        transfer = {
            "from": from_player,
            "to": to_player,
            "amount": amount,
            "round": round_num,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.wealth_transfers.append(transfer)
        
        self.record_event(
            self.EVENT_TRANSFER,
            from_player=from_player,
            to_player=to_player,
            amount=amount,
            round=round_num,
            reason=reason
        )
        
        # Update player wealth if available
        if from_player in self.player_wealth and to_player in self.player_wealth:
            from_prev = self.player_wealth[from_player][-1]["wealth"] if self.player_wealth[from_player] else 0
            to_prev = self.player_wealth[to_player][-1]["wealth"] if self.player_wealth[to_player] else 0
            
            # Update sender's wealth
            self.update_player_wealth(
                from_player, 
                from_prev - amount, 
                round_num, 
                f"transfer_to_{to_player}"
            )
            
            # Update receiver's wealth
            self.update_player_wealth(
                to_player, 
                to_prev + amount, 
                round_num, 
                f"transfer_from_{from_player}"
            )
    
    def record_vote(self, voter: str, voted_for: str, round_num: int) -> None:
        """
        Record a vote cast by a player.
        
        Args:
            voter (str): Player casting the vote
            voted_for (str): Player being voted for
            round_num (int): Round number
        """
        vote = {
            "voter": voter,
            "voted_for": voted_for,
            "round": round_num,
            "timestamp": datetime.now().isoformat()
        }
        
        self.votes.append(vote)
        
        self.record_event(
            self.EVENT_VOTE,
            voter=voter,
            voted_for=voted_for,
            round=round_num
        )
        
        # Add to game summary
        self.game_summary.append(f"Round {round_num}: {voter} voted for {voted_for}")
    
    def record_elimination(self, player: str, round_num: int, wealth: int) -> None:
        """
        Record a player's elimination.
        
        Args:
            player (str): Player being eliminated
            round_num (int): Round number
            wealth (int): Final wealth of the player
        """
        elimination = {
            "player": player,
            "round": round_num,
            "wealth": wealth,
            "timestamp": datetime.now().isoformat()
        }
        
        self.eliminations.append(elimination)
        
        self.record_event(
            self.EVENT_ELIMINATION,
            player=player,
            round=round_num,
            wealth=wealth
        )
        
        # Add to game summary
        self.game_summary.append(f"Round {round_num}: {player} was eliminated with {wealth} wealth")
    
    def evaluate_game(self) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of the entire game.
        
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
        
        # Format game summary
        game_summary_text = "\n".join(self.game_summary)
        
        # Format initial wealth for display
        initial_wealth_text = ", ".join([f"{player}: {wealth}" for player, wealth in self.initial_wealth.items()])
        
        # Format final wealth for display
        final_wealth = {}
        for player, wealth_history in self.player_wealth.items():
            if wealth_history:
                final_wealth[player] = wealth_history[-1]["wealth"]
            
        final_wealth_text = ", ".join([f"{player}: {wealth}" for player, wealth in final_wealth.items()])
        
        # Context for evaluation
        context = {
            "initial_wealth": initial_wealth_text,
            "final_wealth": final_wealth_text,
            "game_summary": game_summary_text,
            "eliminations": len(self.eliminations)
        }
        
        # Request evaluation using the Beast-specific template
        return self.record_llm_evaluation("game", context, self.BEAST_GAME_EVALUATION_TEMPLATE)
    
    def evaluate_conversation(self, player1: str, player2: str, messages: List[Dict[str, Any]], 
                             round_num: int, transfer_outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific conversation.
        
        Args:
            player1 (str): First player in conversation
            player2 (str): Second player in conversation
            messages (List[Dict[str, Any]]): Messages exchanged
            round_num (int): Round number
            transfer_outcome (Dict[str, Any]): Wealth transfer outcome
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
        
        # Get player wealth before conversation
        wealth_before = {}
        for player in [player1, player2]:
            for record in self.player_wealth.get(player, []):
                if record["round"] == round_num:
                    wealth_before[player] = record["wealth"]
                    break
        
        wealth_before_text = ", ".join([f"{player}: {wealth}" for player, wealth in wealth_before.items()])
        
        # Format transcript
        transcript = []
        for msg in messages:
            if isinstance(msg, dict) and "speaker" in msg and "message" in msg:
                transcript.append(f"{msg['speaker']}: {msg['message']}")
            elif isinstance(msg, str):
                transcript.append(msg)
        
        transcript_text = "\n".join(transcript)
        
        # Format transfer outcome
        transfer_text = f"{transfer_outcome['from']} transferred {transfer_outcome['amount']} to {transfer_outcome['to']}"
        
        # Context for evaluation
        context = {
            "players": f"{player1}, {player2}",
            "wealth_before": wealth_before_text,
            "transcript": transcript_text,
            "transfer_outcome": transfer_text,
            "round": round_num
        }
        
        # Request evaluation using the Beast-specific template
        return self.record_llm_evaluation("conversation", context, self.BEAST_CONVERSATION_EVALUATION_TEMPLATE)
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all Beast metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        # Call the parent method to get common metrics
        base_metrics = super().compute_all()
        
        # Add Beast-specific metrics
        beast_metrics = {
            "wealth_metrics": self._compute_wealth_metrics(),
            "conversation_metrics": self._compute_conversation_metrics(),
            "transfer_metrics": self._compute_transfer_metrics(),
            "voting_metrics": self._compute_voting_metrics(),
            "elimination_metrics": self._compute_elimination_metrics(),
            "game_outcome": self._compute_game_outcome()
        }
        
        # Merge metrics
        self.computed_metrics.update(beast_metrics)
        
        return self.computed_metrics
    
    def _compute_wealth_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to player wealth.
        
        Returns:
            Dict[str, Any]: Wealth-related metrics
        """
        if not self.player_wealth:
            return {"count": 0}
        
        # Get final wealth for each player
        final_wealth = {}
        max_round = 0
        for player, wealth_history in self.player_wealth.items():
            if wealth_history:
                final_wealth[player] = wealth_history[-1]["wealth"]
                max_round = max(max_round, wealth_history[-1]["round"])
        
        # Calculate wealth changes
        wealth_changes = {}
        for player, wealth_history in self.player_wealth.items():
            if len(wealth_history) > 1:
                initial = wealth_history[0]["wealth"]
                final = wealth_history[-1]["wealth"]
                wealth_changes[player] = {
                    "initial": initial,
                    "final": final, 
                    "change": final - initial,
                    "percent_change": (final - initial) / max(1, initial) * 100
                }
        
        # Calculate wealth distribution statistics
        initial_values = list(self.initial_wealth.values())
        final_values = list(final_wealth.values())
        
        # Calculate inequality metrics (Gini coefficient simplified)
        def gini(wealth_values):
            sorted_wealth = sorted(wealth_values)
            n = len(sorted_wealth)
            if n == 0:
                return 0
            cumsum = 0
            for i, wealth in enumerate(sorted_wealth):
                cumsum += wealth * (n - i)
            return (2 * cumsum) / (n * sum(sorted_wealth)) - (n + 1) / n
        
        initial_gini = gini(initial_values) if initial_values else 0
        final_gini = gini(final_values) if final_values else 0
        gini_change = final_gini - initial_gini
        
        return {
            "final_wealth": final_wealth,
            "wealth_changes": wealth_changes,
            "inequality": {
                "initial_gini": initial_gini,
                "final_gini": final_gini,
                "gini_change": gini_change
            },
            "wealth_history": self.player_wealth,
            "rounds": max_round
        }
    
    def _compute_conversation_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to player conversations.
        
        Returns:
            Dict[str, Any]: Conversation-related metrics
        """
        if not self.conversations:
            return {"count": 0}
        
        # Count conversations per player
        conversations_per_player = {}
        for conv in self.conversations:
            for player in conv["players"]:
                if player not in conversations_per_player:
                    conversations_per_player[player] = 0
                conversations_per_player[player] += 1
        
        # Count successful transfers (resulting in wealth transfer)
        successful_conversations = [conv for conv in self.conversations if conv.get("transfer_outcome")]
        success_rate = len(successful_conversations) / len(self.conversations) if self.conversations else 0
        
        # Analyze message lengths
        message_lengths = []
        for conv in self.conversations:
            for msg in conv.get("messages", []):
                if isinstance(msg, dict) and "message" in msg:
                    message_lengths.append(len(msg["message"]))
                elif isinstance(msg, str):
                    message_lengths.append(len(msg))
        
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        return {
            "count": len(self.conversations),
            "by_player": conversations_per_player,
            "success_rate": success_rate,
            "avg_message_length": avg_message_length,
            "successful_count": len(successful_conversations)
        }
    
    def _compute_transfer_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to wealth transfers.
        
        Returns:
            Dict[str, Any]: Transfer-related metrics
        """
        if not self.wealth_transfers:
            return {"count": 0}
        
        # Calculate total transfers per player (sent and received)
        sent_per_player = {}
        received_per_player = {}
        
        for transfer in self.wealth_transfers:
            from_player = transfer["from"]
            to_player = transfer["to"]
            amount = transfer["amount"]
            
            if from_player not in sent_per_player:
                sent_per_player[from_player] = 0
            sent_per_player[from_player] += amount
            
            if to_player not in received_per_player:
                received_per_player[to_player] = 0
            received_per_player[to_player] += amount
            
        # Calculate net transfers
        net_transfers = {}
        all_players = set(list(sent_per_player.keys()) + list(received_per_player.keys()))
        
        for player in all_players:
            sent = sent_per_player.get(player, 0)
            received = received_per_player.get(player, 0)
            net_transfers[player] = received - sent
        
        # Calculate transfer statistics
        transfer_amounts = [t["amount"] for t in self.wealth_transfers]
        avg_transfer = sum(transfer_amounts) / len(transfer_amounts) if transfer_amounts else 0
        max_transfer = max(transfer_amounts) if transfer_amounts else 0
        
        # Group by reason
        by_reason = {}
        for transfer in self.wealth_transfers:
            reason = transfer.get("reason", "unknown")
            if reason not in by_reason:
                by_reason[reason] = 0
            by_reason[reason] += 1
        
        return {
            "count": len(self.wealth_transfers),
            "sent_per_player": sent_per_player,
            "received_per_player": received_per_player,
            "net_transfers": net_transfers,
            "avg_transfer": avg_transfer,
            "max_transfer": max_transfer,
            "by_reason": by_reason,
            "total_transferred": sum(transfer_amounts),
            "transfers": self.wealth_transfers
        }
    
    def _compute_voting_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to voting.
        
        Returns:
            Dict[str, Any]: Voting-related metrics
        """
        if not self.votes:
            return {"count": 0}
        
        # Count votes by round
        votes_by_round = {}
        for vote in self.votes:
            round_num = vote["round"]
            if round_num not in votes_by_round:
                votes_by_round[round_num] = 0
            votes_by_round[round_num] += 1
        
        # Count votes received per player
        votes_received = {}
        for vote in self.votes:
            voted_for = vote["voted_for"]
            if voted_for not in votes_received:
                votes_received[voted_for] = 0
            votes_received[voted_for] += 1
        
        # Calculate most voted players
        most_voted = []
        if votes_received:
            max_votes = max(votes_received.values())
            most_voted = [player for player, count in votes_received.items() if count == max_votes]
        
        # Check correlation between wealth and votes
        wealth_vote_correlation = 0
        if self.player_wealth and votes_received:
            # Simplified correlation measure - would be more sophisticated in production
            common_players = set(self.player_wealth.keys()) & set(votes_received.keys())
            if common_players:
                # Higher wealth tends to get more votes?
                wealthy_players = sorted(common_players, 
                                       key=lambda p: self.player_wealth[p][-1]["wealth"] if self.player_wealth[p] else 0,
                                       reverse=True)
                most_voted_players = sorted(common_players,
                                          key=lambda p: votes_received.get(p, 0),
                                          reverse=True)
                
                # Simple rank correlation proxy
                correlation_sum = 0
                for player in common_players:
                    wealth_rank = wealthy_players.index(player)
                    vote_rank = most_voted_players.index(player)
                    correlation_sum += abs(wealth_rank - vote_rank)
                
                # Normalize to 0-1 range (1 means perfect correlation)
                max_diff_sum = len(common_players) * len(common_players)
                wealth_vote_correlation = 1 - (correlation_sum / max_diff_sum if max_diff_sum > 0 else 0)
        
        return {
            "count": len(self.votes),
            "by_round": votes_by_round,
            "votes_received": votes_received,
            "most_voted": most_voted,
            "wealth_vote_correlation": wealth_vote_correlation,
            "votes": self.votes
        }
    
    def _compute_elimination_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to player eliminations.
        
        Returns:
            Dict[str, Any]: Elimination-related metrics
        """
        if not self.eliminations:
            return {"count": 0}
        
        # Calculate elimination order
        elimination_order = [e["player"] for e in sorted(self.eliminations, key=lambda x: x["round"])]
        
        # Calculate wealth at elimination
        elimination_wealth = {e["player"]: e["wealth"] for e in self.eliminations}
        
        # Calculate rounds to elimination
        elimination_rounds = {e["player"]: e["round"] for e in self.eliminations}
        
        return {
            "count": len(self.eliminations),
            "order": elimination_order,
            "wealth": elimination_wealth,
            "rounds": elimination_rounds,
            "eliminations": self.eliminations
        }
    
    def _compute_game_outcome(self) -> Dict[str, Any]:
        """
        Compute metrics related to the game outcome.
        
        Returns:
            Dict[str, Any]: Outcome-related metrics
        """
        # Get final wealth for non-eliminated players
        remaining_players = {}
        eliminated_players = [e["player"] for e in self.eliminations]
        
        for player, wealth_history in self.player_wealth.items():
            if player not in eliminated_players and wealth_history:
                remaining_players[player] = wealth_history[-1]["wealth"]
        
        # Determine winner (player with most wealth among remaining)
        winner = None
        max_wealth = -1
        for player, wealth in remaining_players.items():
            if wealth > max_wealth:
                max_wealth = wealth
                winner = player
                
        # Calculate wealth inequality among remaining players
        remaining_wealth = list(remaining_players.values())
        
        # Calculate Gini coefficient for remaining players
        def gini(wealth_values):
            sorted_wealth = sorted(wealth_values)
            n = len(sorted_wealth)
            if n <= 1:
                return 0
            cumsum = 0
            for i, wealth in enumerate(sorted_wealth):
                cumsum += wealth * (n - i)
            return (2 * cumsum) / (n * sum(sorted_wealth)) - (n + 1) / n
        
        remaining_gini = gini(remaining_wealth) if remaining_wealth else 0
        
        return {
            "remaining_players": remaining_players,
            "winner": winner,
            "winner_wealth": max_wealth if winner else 0,
            "remaining_gini": remaining_gini,
            "eliminated_count": len(eliminated_players),
            "remaining_count": len(remaining_players)
        } 