from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os

class BaseMetrics(ABC):
    """
    Abstract base class for metrics collection and computation across all games.
    
    This class provides a unified interface for collecting metrics during games,
    computing final metric values, and exporting results to different formats.
    Each game should extend this class with game-specific metrics.
    
    Attributes:
        events (List[Dict[str, Any]]): Chronological list of all recorded events.
        metadata (Dict[str, Any]): Game metadata like game type, model names, etc.
        computed_metrics (Dict[str, Any]): Final computed metric values.
    """
    
    # Common event types
    EVENT_GAME_START = "game_start"
    EVENT_GAME_END = "game_end"
    EVENT_ROUND_START = "round_start"
    EVENT_ROUND_END = "round_end"
    EVENT_TURN_START = "turn_start"
    EVENT_TURN_END = "turn_end"
    EVENT_MODEL_REQUEST = "model_request"
    EVENT_MODEL_RESPONSE = "model_response"
    EVENT_DECISION = "decision"
    EVENT_ACTION = "action"
    EVENT_LLM_EVALUATION = "llm_evaluation"
    
    def __init__(self, game_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics collector.
        
        Args:
            game_type (str): Type of the game (spyfall, beast, askguess, etc.)
            metadata (Optional[Dict[str, Any]]): Additional metadata for the game session
        """
        self.game_type = game_type
        self.events: List[Dict[str, Any]] = []
        self.metadata = metadata or {}
        self.metadata["game_type"] = game_type
        self.metadata["timestamp"] = datetime.now().isoformat()
        self.computed_metrics: Dict[str, Any] = {}
        self.llm_evaluator = None
        self.use_llm_evaluation = False
    
    def enable_llm_evaluation(self, llm_model: Any) -> None:
        """
        Enable LLM evaluation of the game using the provided model.
        
        Args:
            llm_model: LLM model instance to use for evaluation
        """
        self.llm_evaluator = llm_model
        self.use_llm_evaluation = True
        self.add_metadata("llm_evaluation_enabled", True)
        self.add_metadata("llm_evaluator_model", str(llm_model.__class__.__name__))
        
    def record_llm_evaluation(self, phase: str, context: Dict[str, Any], prompt_template: str = None) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific game phase.
        
        Args:
            phase (str): Game phase being evaluated (e.g., "round", "turn", "game")
            context (Dict[str, Any]): Context information about the game state
            prompt_template (str, optional): Custom prompt template for the evaluation
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation or not self.llm_evaluator:
            return None
            
        # Default prompt templates for different phases
        templates = {
            "game": """
                Evaluate this game of {game_type} based on the provided information:
                
                Game context: {context}
                
                Please analyze the following aspects:
                1. Strategy quality of each player
                2. Effectiveness of communication
                3. Key turning points in the game
                4. Overall fairness and balance
                
                Provide a structured evaluation with scores (1-10) for each aspect
                and a brief explanation.
            """,
            "round": """
                Evaluate this round of {game_type} based on the provided information:
                
                Round context: {context}
                
                Please analyze:
                1. Key decisions made by players
                2. Quality of deception and detection
                3. Most influential actions
                
                Provide scores (1-10) for each player's performance this round
                and brief reasoning.
            """,
            "turn": """
                Evaluate this player turn in {game_type} based on the provided information:
                
                Turn context: {context}
                
                Please analyze:
                1. Quality of the player's decision/action
                2. Strategic implications
                3. Effectiveness of communication
                
                Score this turn (1-10) and provide brief reasoning.
            """
        }
        
        # Use provided template or default
        prompt = prompt_template or templates.get(phase, templates["turn"])
        
        # Get all keys from context to check if template keys are available
        context_keys = set(context.keys())
        
        # Format context as JSON
        context_json = json.dumps(context, indent=2)
        
        try:
            # First try format with the basic game_type and context JSON
            base_context = {
                "game_type": self.game_type,
                "context": context_json
            }
            
            # Add all context items as top-level keys for formatting
            format_context = base_context.copy()
            format_context.update(context)
            
            # Format prompt with context
            formatted_prompt = prompt.format(**format_context)
            
            # Send to LLM for evaluation
            # Support for LangChain models which use invoke method
            from langchain_core.messages import HumanMessage, SystemMessage
            
            evaluation_result = None
            
            # Try different methods to handle various LLM interfaces
            try:
                # Method 1: Using LangChain invoke with messages
                messages = [
                    SystemMessage(content="You are a helpful AI evaluator tasked with analyzing game situations."),
                    HumanMessage(content=formatted_prompt)
                ]
                evaluation_result = self.llm_evaluator.invoke(messages)
                
                # Extract content from LangChain response object if needed
                if hasattr(evaluation_result, 'content'):
                    evaluation = evaluation_result.content
                else:
                    evaluation = str(evaluation_result)
                    
            except (AttributeError, TypeError) as e:
                # Method 2: Try direct invoke with string
                try:
                    evaluation_result = self.llm_evaluator.invoke(formatted_prompt)
                    evaluation = str(evaluation_result)
                except (AttributeError, TypeError):
                    # Method 3: Fallback to legacy chat/generate methods if available
                    if hasattr(self.llm_evaluator, "generate"):
                        evaluation = self.llm_evaluator.generate([formatted_prompt]).generations[0][0].text
                    elif hasattr(self.llm_evaluator, "chat"):
                        evaluation = self.llm_evaluator.chat(formatted_prompt)
                    else:
                        # Final fallback - attempt direct call
                        evaluation = str(self.llm_evaluator(formatted_prompt))
                
            # Parse the response
            result = {
                "phase": phase,
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Record the evaluation event
            self.record_event(
                self.EVENT_LLM_EVALUATION,
                phase=phase,
                context=context,
                evaluation=evaluation
            )
            
            return result
            
        except KeyError as e:
            # Handle missing template keys gracefully
            print(f"Warning: Missing key {e} in LLM evaluation template. Using simplified prompt.")
            
            # Fall back to a simplified template
            simplified_prompt = f"""
                Evaluate this {phase} in {self.game_type} based on the provided information:
                
                Context: {context_json}
                
                Please provide a thorough analysis with scores (1-10) where appropriate.
            """
            
            try:
                # Try different methods for the simplified prompt
                try:
                    messages = [
                        SystemMessage(content="You are a helpful AI evaluator tasked with analyzing game situations."),
                        HumanMessage(content=simplified_prompt)
                    ]
                    evaluation_result = self.llm_evaluator.invoke(messages)
                    
                    if hasattr(evaluation_result, 'content'):
                        evaluation = evaluation_result.content
                    else:
                        evaluation = str(evaluation_result)
                        
                except (AttributeError, TypeError):
                    try:
                        evaluation_result = self.llm_evaluator.invoke(simplified_prompt)
                        evaluation = str(evaluation_result)
                    except (AttributeError, TypeError):
                        if hasattr(self.llm_evaluator, "generate"):
                            evaluation = self.llm_evaluator.generate([simplified_prompt]).generations[0][0].text
                        elif hasattr(self.llm_evaluator, "chat"):
                            evaluation = self.llm_evaluator.chat(simplified_prompt) 
                        else:
                            evaluation = str(self.llm_evaluator(simplified_prompt))
                
                result = {
                    "phase": phase,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "used_fallback": True
                }
                
                self.record_event(
                    self.EVENT_LLM_EVALUATION,
                    phase=phase,
                    context=context,
                    evaluation=evaluation,
                    used_fallback=True
                )
                
                return result
                
            except Exception as inner_e:
                print(f"Error in LLM evaluation fallback: {inner_e}")
                return {
                    "phase": phase,
                    "error": str(inner_e),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            # Log error but don't halt game execution
            print(f"Error in LLM evaluation: {e}")
            return {
                "phase": phase,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def _compute_llm_evaluations(self) -> Dict[str, Any]:
        """
        Compile and analyze all LLM evaluations from the game.
        
        Returns:
            Dict[str, Any]: Compiled LLM evaluation metrics
        """
        if not self.use_llm_evaluation:
            return {"enabled": False}
            
        eval_events = [e for e in self.events if e["type"] == self.EVENT_LLM_EVALUATION]
        
        if not eval_events:
            return {"enabled": True, "evaluations_count": 0}
            
        evaluations_by_phase = {}
        
        # Group evaluations by phase
        for event in eval_events:
            phase = event["data"].get("phase", "unknown")
            
            if phase not in evaluations_by_phase:
                evaluations_by_phase[phase] = []
                
            evaluations_by_phase[phase].append({
                "timestamp": event["timestamp"],
                "evaluation": event["data"].get("evaluation", ""),
                "context": event["data"].get("context", {})
            })
        
        return {
            "enabled": True,
            "evaluations_count": len(eval_events),
            "by_phase": evaluations_by_phase
        }
    
    def record_event(self, event_type: str, **data) -> None:
        """
        Record a game event with timestamp and associated data.
        
        Args:
            event_type (str): Type of the event (see class constants)
            **data: Additional data about the event
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.events.append(event)
    
    def record_model_interaction(self, agent_name: str, request: str, response: str, 
                                model_name: str, tokens_in: int = None, 
                                tokens_out: int = None, latency: float = None) -> None:
        """
        Record an LLM request/response interaction with detailed metrics.
        
        Args:
            agent_name (str): Name of the agent making the request
            request (str): The prompt sent to the model
            response (str): The model's response
            model_name (str): Name of the model used
            tokens_in (int, optional): Number of input tokens
            tokens_out (int, optional): Number of output tokens
            latency (float, optional): Response time in seconds
        """
        self.record_event(
            self.EVENT_MODEL_REQUEST,
            agent=agent_name,
            model=model_name,
            prompt=request,
            tokens=tokens_in
        )
        
        self.record_event(
            self.EVENT_MODEL_RESPONSE,
            agent=agent_name,
            model=model_name,
            response=response,
            tokens=tokens_out,
            latency=latency
        )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update metadata.
        
        Args:
            key (str): Metadata key
            value (Any): Value to store
        """
        self.metadata[key] = value
    
    def record_description(self, player_name: str, description: str, is_spy: bool) -> None:
        """
        Record a player's description.
        
        Args:
            player_name (str): Name of the player
            description (str): The description given
            is_spy (bool): Whether the player is a spy
        """
        self.record_event(
            "description",
            agent=player_name,
            content=description,
            is_spy=is_spy
        )
    
    def record_vote(self, player_name: str, voted_for: str, is_spy: bool, reasoning: str = "") -> None:
        """
        Record a player's vote.
        
        Args:
            player_name (str): Name of the player voting
            voted_for (str): Name of the player voted for
            is_spy (bool): Whether the voting player is a spy
            reasoning (str): Reasoning behind the vote
        """
        self.record_event(
            "vote",
            agent=player_name,
            voted_for=voted_for,
            is_spy=is_spy,
            reasoning=reasoning
        )
    
    def record_role_assignment(self, players: List[str], spy_index: int, spy_name: str) -> None:
        """
        Record role assignments for the game.
        
        Args:
            players (List[str]): List of all player names
            spy_index (int): Index of the spy
            spy_name (str): Name of the spy
        """
        self.record_event(
            "role_assignment",
            players=players,
            spy_index=spy_index,
            spy_name=spy_name
        )
        
        # Add to metadata for easy access
        self.add_metadata("players", players)
        self.add_metadata("spy_index", spy_index)
        self.add_metadata("spy_name", spy_name)
    
    def add_game_words(self, spy_word: str, villager_word: str) -> None:
        """
        Add game words to metadata.
        
        Args:
            spy_word (str): The word given to the spy
            villager_word (str): The word given to villagers
        """
        self.add_metadata("spy_word", spy_word)
        self.add_metadata("villager_word", villager_word)
    
    def record_game_end(self, winner: str, spy_caught: bool) -> None:
        """
        Record the end of the game.
        
        Args:
            winner (str): The winning role ("spy", "villager", "error")
            spy_caught (bool): Whether the spy was caught
        """
        self.record_event(
            self.EVENT_GAME_END,
            winner=winner,
            spy_caught=spy_caught
        )
        
        # Add to metadata for easy access
        self.add_metadata("winner", winner)
        self.add_metadata("spy_caught", spy_caught)
    
    def evaluate_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Evaluate a round using LLM judge.
        
        Args:
            round_number (int): Round number to evaluate
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None
        """
        if not self.use_llm_evaluation or not self.llm_evaluator:
            return None
            
        # Get round events
        round_events = [e for e in self.events if 
                        e.get("data", {}).get("round_number") == round_number]
        
        # Context for evaluation
        context = {
            "round_number": round_number,
            "events": round_events,
            "metadata": {k: v for k, v in self.metadata.items() 
                         if k in ["spy_word", "villager_word", "spy_name"]}
        }
        
        return self.record_llm_evaluation("round", context)
    
    def evaluate_game(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate the entire game using LLM judge.
        
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None
        """
        if not self.use_llm_evaluation or not self.llm_evaluator:
            return None
            
        # Context for evaluation
        context = {
            "events": self.events,
            "metadata": self.metadata
        }
        
        return self.record_llm_evaluation("game", context)
    
    @abstractmethod
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all metrics based on collected events.
        
        This should be implemented by each game to calculate game-specific metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        # Basic metrics that apply to all games
        self.computed_metrics = {
            "metadata": self.metadata,
            "timing": self._compute_timing_metrics(),
            "model_usage": self._compute_model_usage(),
            "interaction": self._compute_interaction_metrics(),
            "llm_evaluations": self._compute_llm_evaluations()
        }
        return self.computed_metrics
    
    def _compute_timing_metrics(self) -> Dict[str, Any]:
        """
        Compute timing-related metrics.
        
        Returns:
            Dict[str, Any]: Timing metrics including game duration, turns duration, etc.
        """
        timing = {}
        
        # Game duration
        game_start = next((e for e in self.events if e["type"] == self.EVENT_GAME_START), None)
        game_end = next((e for e in self.events if e["type"] == self.EVENT_GAME_END), None)
        
        if game_start and game_end:
            start_time = datetime.fromisoformat(game_start["timestamp"])
            end_time = datetime.fromisoformat(game_end["timestamp"])
            timing["total_duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Round durations
        round_durations = []
        round_start_events = [e for e in self.events if e["type"] == self.EVENT_ROUND_START]
        round_end_events = [e for e in self.events if e["type"] == self.EVENT_ROUND_END]
        
        for start, end in zip(round_start_events, round_end_events):
            start_time = datetime.fromisoformat(start["timestamp"])
            end_time = datetime.fromisoformat(end["timestamp"])
            round_durations.append((end_time - start_time).total_seconds())
            
        if round_durations:
            timing["rounds_count"] = len(round_durations)
            timing["avg_round_duration_seconds"] = sum(round_durations) / len(round_durations)
            timing["min_round_duration_seconds"] = min(round_durations)
            timing["max_round_duration_seconds"] = max(round_durations)
        
        # Turn durations by agent
        turn_durations_by_agent = {}
        
        turn_start_events = [e for e in self.events if e["type"] == self.EVENT_TURN_START]
        turn_end_events = [e for e in self.events if e["type"] == self.EVENT_TURN_END]
        
        for start, end in zip(turn_start_events, turn_end_events):
            if "agent" not in start["data"] or "agent" not in end["data"]:
                continue
                
            agent = start["data"]["agent"]
            start_time = datetime.fromisoformat(start["timestamp"])
            end_time = datetime.fromisoformat(end["timestamp"])
            duration = (end_time - start_time).total_seconds()
            
            if agent not in turn_durations_by_agent:
                turn_durations_by_agent[agent] = []
                
            turn_durations_by_agent[agent].append(duration)
        
        agent_timing = {}
        for agent, durations in turn_durations_by_agent.items():
            agent_timing[agent] = {
                "turns_count": len(durations),
                "avg_turn_duration_seconds": sum(durations) / len(durations),
                "min_turn_duration_seconds": min(durations),
                "max_turn_duration_seconds": max(durations)
            }
            
        if agent_timing:
            timing["agent_timing"] = agent_timing
            
        return timing
    
    def _compute_model_usage(self) -> Dict[str, Any]:
        """
        Compute model usage metrics.
        
        Returns:
            Dict[str, Any]: Model usage metrics including token counts, latencies, etc.
        """
        request_events = [e for e in self.events if e["type"] == self.EVENT_MODEL_REQUEST]
        response_events = [e for e in self.events if e["type"] == self.EVENT_MODEL_RESPONSE]
        
        model_usage = {}
        
        # Aggregate by model
        model_data = {}
        for req, resp in zip(request_events, response_events):
            if "model" not in req["data"] or "model" not in resp["data"]:
                continue
                
            model = req["data"]["model"]
            
            if model not in model_data:
                model_data[model] = {
                    "requests_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_latency": 0,
                    "latencies": []
                }
            
            model_data[model]["requests_count"] += 1
            
            if "tokens" in req["data"] and req["data"]["tokens"] is not None:
                model_data[model]["total_input_tokens"] += req["data"]["tokens"]
                
            if "tokens" in resp["data"] and resp["data"]["tokens"] is not None:
                model_data[model]["total_output_tokens"] += resp["data"]["tokens"]
                
            if "latency" in resp["data"]:
                model_data[model]["total_latency"] += resp["data"]["latency"]
                model_data[model]["latencies"].append(resp["data"]["latency"])
        
        # Calculate averages and other stats
        for model, data in model_data.items():
            if data["requests_count"] > 0:
                data["avg_input_tokens"] = data["total_input_tokens"] / data["requests_count"]
                data["avg_output_tokens"] = data["total_output_tokens"] / data["requests_count"]
                
            if data["latencies"]:
                data["avg_latency"] = data["total_latency"] / len(data["latencies"])
                data["min_latency"] = min(data["latencies"])
                data["max_latency"] = max(data["latencies"])
                
            # Remove the raw latencies list to keep the output clean
            data.pop("latencies", None)
        
        model_usage["by_model"] = model_data
        
        # Aggregate by agent
        agent_data = {}
        for req, resp in zip(request_events, response_events):
            if "agent" not in req["data"] or "agent" not in resp["data"]:
                continue
                
            agent = req["data"]["agent"]
            
            if agent not in agent_data:
                agent_data[agent] = {
                    "requests_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0
                }
            
            agent_data[agent]["requests_count"] += 1
            
            if "tokens" in req["data"] and req["data"]["tokens"] is not None:
                agent_data[agent]["total_input_tokens"] += req["data"]["tokens"]
                
            if "tokens" in resp["data"] and resp["data"]["tokens"] is not None:
                agent_data[agent]["total_output_tokens"] += resp["data"]["tokens"]
        
        # Calculate averages
        for agent, data in agent_data.items():
            if data["requests_count"] > 0:
                data["avg_input_tokens"] = data["total_input_tokens"] / data["requests_count"]
                data["avg_output_tokens"] = data["total_output_tokens"] / data["requests_count"]
        
        model_usage["by_agent"] = agent_data
        
        # Overall totals
        model_usage["total_requests"] = sum(data["requests_count"] for data in model_data.values())
        model_usage["total_input_tokens"] = sum(data["total_input_tokens"] for data in model_data.values())
        model_usage["total_output_tokens"] = sum(data["total_output_tokens"] for data in model_data.values())
        model_usage["total_tokens"] = model_usage["total_input_tokens"] + model_usage["total_output_tokens"]
        
        return model_usage
    
    def _compute_interaction_metrics(self) -> Dict[str, Any]:
        """
        Compute interaction metrics.
        
        Returns:
            Dict[str, Any]: Interaction metrics including message counts, lengths, etc.
        """
        interaction = {}
        
        # Action events by agent
        action_events = [e for e in self.events if e["type"] == self.EVENT_ACTION]
        agents = set()
        
        for event in action_events:
            if "agent" in event["data"]:
                agents.add(event["data"]["agent"])
        
        # Count actions per agent
        action_counts = {}
        for agent in agents:
            agent_actions = [e for e in action_events if e["data"].get("agent") == agent]
            action_counts[agent] = len(agent_actions)
        
        interaction["action_counts"] = action_counts
        interaction["total_actions"] = len(action_events)
        
        # Decision events
        decision_events = [e for e in self.events if e["type"] == self.EVENT_DECISION]
        interaction["total_decisions"] = len(decision_events)
        
        return interaction
    
    def to_json(self) -> str:
        """
        Convert metrics to JSON string.
        
        Returns:
            str: JSON representation of metrics
        """
        result = {
            "metadata": self.metadata,
            "metrics": self.computed_metrics,
            "events": self.events
        }
        return json.dumps(result, indent=2)
    
    def save(self, file_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            file_path (str): Path to save the metrics file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Ensure we have computed metrics
        if not self.computed_metrics:
            self.compute_all()
            
        # Write to file
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: str) -> 'BaseMetrics':
        """
        Load metrics from a JSON file.
        
        Args:
            file_path (str): Path to the metrics file
            
        Returns:
            BaseMetrics: Loaded metrics object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        game_type = data.get("metadata", {}).get("game_type", "unknown")
        instance = cls(game_type, data.get("metadata", {}))
        
        instance.events = data.get("events", [])
        instance.computed_metrics = data.get("metrics", {})
        
        return instance 