from typing import Dict, List, Tuple, Union, Any, Set
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class SpyfallMetrics:
    """Metrics for evaluating Spyfall game performance."""
    
    @staticmethod
    def spy_survival_rate(results: List[Dict[str, Any]]) -> float:
        """
        Calculate the percentage of games where the spy survived.
        
        Args:
            results: List of game result dictionaries
            
        Returns:
            Survival rate as a float between 0 and 1
        """
        spy_games = [r for r in results if r.get("spy_id") is not None]
        if not spy_games:
            return 0.0
            
        survived = sum(1 for r in spy_games if not r.get("spy_caught", False))
        return survived / len(spy_games)
    
    @staticmethod
    def villager_detection_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate F1 score for villagers' ability to detect spies.
        
        Args:
            results: List of game result dictionaries
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        y_true = []
        y_pred = []
        
        for result in results:
            spy_id = result.get("spy_id")
            if spy_id is None:
                continue
                
            spy_name = result.get("players", [])[spy_id-1] if "players" in result and 0 <= spy_id-1 < len(result.get("players", [])) else None
            
            votes = result.get("votes", {})
            for player_id, vote in votes.items():
                # Skip spy's vote
                try:
                    player_id_int = int(player_id)
                    is_spy_vote = player_id_int == spy_id
                except ValueError:
                    # Если player_id не число, сравниваем по имени
                    is_spy_vote = player_id == spy_name
                
                if is_spy_vote:
                    continue
                
                # Check if vote is for spy by comparing either ID or name
                is_correct_vote = False
                try:
                    vote_int = int(vote)
                    is_correct_vote = vote_int == spy_id
                except ValueError:
                    # Если vote не число, сравниваем по имени
                    is_correct_vote = vote == spy_name
                
                # Add to evaluation data
                y_true.append(1 if is_correct_vote else 0)
                y_pred.append(1)  # Villager voted for someone
        
        if not y_true:
            return {"precision": 0, "recall": 0, "f1": 0}
            
        precision = sum(y_true) / len(y_true) if y_true else 0
        recall = sum(y_true) / sum(1 for r in results if r.get("spy_id") is not None)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1": f1
        }
    
    @staticmethod
    def matthews_correlation(results: List[Dict[str, Any]]) -> float:
        """
        Calculate Matthews Correlation Coefficient for spy detection.
        
        Args:
            results: List of game result dictionaries
            
        Returns:
            MCC score as a float between -1 and 1
        """
        y_true = []
        y_pred = []
        
        for result in results:
            spy_id = result.get("spy_id")
            if spy_id is None:
                continue
                
            spy_name = result.get("players", [])[spy_id-1] if "players" in result and 0 <= spy_id-1 < len(result.get("players", [])) else None
            
            votes = result.get("votes", {})
            for player_id, vote in votes.items():
                # Проверяем, является ли vote именем шпиона или его ID
                vote_for_spy = False
                try:
                    vote_int = int(vote)
                    vote_for_spy = vote_int == spy_id
                except ValueError:
                    vote_for_spy = vote == spy_name
                
                # Проверяем, является ли голосующий шпионом
                is_spy = False
                try:
                    player_id_int = int(player_id)
                    is_spy = player_id_int == spy_id
                except ValueError:
                    is_spy = player_id == spy_name
                
                y_true.append(1 if vote_for_spy else 0)
                y_pred.append(1 if is_spy else 0)
        
        if not y_true or len(set(y_true)) < 2 or len(set(y_pred)) < 2:
            return 0.0
            
        return matthews_corrcoef(y_true, y_pred)
    
    @staticmethod
    def top_k_suspect_accuracy(results: List[Dict[str, Any]], k: int = 3) -> float:
        """
        Calculate accuracy when considering the top-k voted players.
        
        Args:
            results: List of game result dictionaries
            k: Number of top suspects to consider
            
        Returns:
            Top-k accuracy as a float between 0 and 1
        """
        correct = 0
        total = 0
        
        for result in results:
            spy_id = result.get("spy_id")
            if spy_id is None:
                continue
                
            spy_name = result.get("players", [])[spy_id-1] if "players" in result and 0 <= spy_id-1 < len(result.get("players", [])) else None
            
            # Count votes for each player
            vote_counts = {}
            for vote in result.get("votes", {}).values():
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
            
            # Get top-k suspects
            top_k = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Проверяем, входит ли шпион в топ-k по имени или ID
            spy_in_top_k = False
            for suspect_id, _ in top_k:
                try:
                    if int(suspect_id) == spy_id:
                        spy_in_top_k = True
                        break
                except ValueError:
                    if suspect_id == spy_name:
                        spy_in_top_k = True
                        break
            
            if spy_in_top_k:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def description_specificity(description: str, abstract_words: Set[str] = None) -> float:
        """
        Calculate how specific a description is by measuring the absence of abstract/vague terms.
        
        Args:
            description: The text description to analyze
            abstract_words: Set of abstract/vague words to check against
            
        Returns:
            Specificity score between 0 and 1 (higher is more specific)
        """
        if abstract_words is None:
            abstract_words = {"device", "thing", "object", "item", "stuff", "something", 
                             "anything", "everything", "nothing", "way", "kind", "sort", 
                             "type", "example", "instance", "case", "part", "piece", 
                             "bit", "element", "feature", "aspect", "factor", "component"}
        
        # Tokenize description
        tokens = re.findall(r'\b\w+\b', description.lower())
        if not tokens:
            return 0.0
            
        # Count abstract words
        abstract_count = sum(1 for token in tokens if token in abstract_words)
        
        # Calculate specificity score
        return 1.0 - (abstract_count / len(tokens)) if len(tokens) > 0 else 0.0
    
    @staticmethod
    def calculate_perplexity(text: str, api_key: str = None, model_name: str = "gpt-2") -> float:
        """
        Calculate perplexity using either OpenAI API or a local approach.
        
        Args:
            text: The text to evaluate
            api_key: OpenAI API key (optional)
            model_name: Model to use (gpt-2, ada, babbage, curie, davinci, etc.)
            
        Returns:
            Perplexity score (lower is more natural)
        """
        if not text:
            return float('inf')
            
        # OpenAI models available for completions with logprobs
        valid_openai_models = ["text-embedding-3-small", "davinci-002", "gpt-4-0613", "gpt-3.5-turbo-instruct"]
        
        # Определяем нужную переменную окружения по имени модели
        if not api_key and any(model in model_name for model in ["gpt", "davinci", "text-embedding", "babbage"]):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(f"Warning: OpenAI model requested but OPENAI_API_KEY not found in environment")
        
        # Проверяем модель и API ключ
        if api_key and model_name in valid_openai_models:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "prompt": text[:100],  # Use first 100 chars as context
                        "max_tokens": 0,
                        "echo": True,
                        "logprobs": 0
                    }
                )
                data = response.json()
                if "choices" in data and data["choices"]:
                    logprobs = data["choices"][0].get("logprobs", {}).get("token_logprobs", [])
                    if logprobs:
                        # Skip first token as it has no context
                        logprobs = logprobs[1:]
                        if logprobs:
                            # Calculate perplexity as exp(-mean(logprobs))
                            return np.exp(-np.mean(logprobs))
            except Exception as e:
                print(f"OpenAI API error: {e}")
                print("Falling back to local perplexity calculation")
        
        # Local perplexity calculation
        try:
            # Try to use transformers if available
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            # Load pre-trained model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Encode text
            inputs = tokenizer(text, return_tensors="pt")
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
            # Perplexity = exp(loss)
            return torch.exp(loss).item()
            
        except (ImportError, Exception) as e:
            print(f"Transformers error: {e}")
            print("Using character-based entropy as fallback")
            
            # Most basic fallback - character frequency model
            char_freq = {}
            for char in text.lower():
                char_freq[char] = char_freq.get(char, 0) + 1
            
            entropy = 0
            total_chars = len(text)
            for count in char_freq.values():
                prob = count / total_chars
                entropy -= prob * np.log2(prob)
            
            return 2 ** entropy  # Perplexity is 2^entropy
    
    @staticmethod
    def vagueness_score(descriptions: List[str]) -> Dict[str, float]:
        """
        Calculate vagueness score based on TF-IDF of rare vs common words.
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            Dictionary mapping each description to its vagueness score
        """
        if not descriptions:
            return {}
            
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9)
        
        try:
            # Transform descriptions to TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            
            # Get feature names (words)
            feature_names = vectorizer.get_feature_names_out()
            
            scores = {}
            for i, desc in enumerate(descriptions):
                # Get non-zero elements in this row
                row = tfidf_matrix[i].toarray()[0]
                
                # Get rare words (high TF-IDF) and common words (low TF-IDF)
                rare_word_scores = sum(score for score in row if score > 0.1)
                common_word_scores = sum(score for score in row if 0 < score <= 0.1)
                
                # Vagueness score is ratio of common to rare words
                if rare_word_scores > 0:
                    scores[desc] = common_word_scores / rare_word_scores
                else:
                    scores[desc] = 1.0  # All common words
                    
            return scores
            
        except Exception as e:
            print(f"Error calculating vagueness: {e}")
            return {desc: 0.0 for desc in descriptions}
    
    @staticmethod
    def brier_score(results: List[Dict[str, Any]]) -> float:
        """
        Calculate Brier score to measure calibration of probability predictions.
        
        Args:
            results: List of game result dictionaries
            
        Returns:
            Brier score (lower is better)
        """
        squared_errors = []
        
        for result in results:
            spy_id = result.get("spy_id")
            if spy_id is None or not result.get("vote_confidences", {}):
                continue
                
            confidences = result.get("vote_confidences", {})
            for player_id, confidence_data in confidences.items():
                for suspect_id, confidence in confidence_data.items():
                    # Binary outcome: 1 if suspect is spy, 0 otherwise
                    actual = 1 if int(suspect_id) == spy_id else 0
                    # Convert percentage confidence to probability [0-1]
                    predicted = min(1.0, max(0.0, confidence / 100))
                    squared_errors.append((predicted - actual) ** 2)
                    
        if not squared_errors:
            return 1.0  # Worst possible score
            
        return sum(squared_errors) / len(squared_errors)
    
    @staticmethod
    def vote_influence_index(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate how much each player influences others' votes.
        
        Args:
            results: List of game result dictionaries
            
        Returns:
            Dictionary mapping player positions to influence scores
        """
        influence_scores = {}
        vote_counts = {}
        
        for result in results:
            if not result.get("vote_sequence", []):
                continue
                
            vote_sequence = result.get("vote_sequence", [])
            for i, (voter_id, vote) in enumerate(vote_sequence):
                # Skip first voter as they can't be influenced
                if i == 0:
                    continue
                    
                # Check how many previous voters voted the same way
                matching_votes = sum(1 for prev_voter, prev_vote in vote_sequence[:i] 
                                    if prev_vote == vote)
                other_votes = i - matching_votes  # Number of previous voters with different votes
                
                # Calculate influence score (matching votes / other votes)
                influence = matching_votes / (other_votes + 1e-10)  # Avoid division by zero
                
                # Track by position in voting sequence
                position = i + 1
                if position not in influence_scores:
                    influence_scores[position] = []
                    vote_counts[position] = 0
                    
                influence_scores[position].append(influence)
                vote_counts[position] += 1
                
        # Calculate average influence score for each position
        avg_influence = {}
        for position, scores in influence_scores.items():
            avg_influence[str(position)] = sum(scores) / len(scores) if scores else 0.0
            
        return avg_influence
    
    @staticmethod
    def cot_coherence(results: List[Dict[str, Any]], 
                       model_type: str = "local",
                       model_name: str = "all-MiniLM-L6-v2", 
                       api_key: str = None) -> Dict[str, float]:
        """
        Calculate coherence between thought process and spoken statement.
        
        Args:
            results: List of game result dictionaries
            model_type: "local" or "openai" 
            model_name: Name of model to use (sentence transformer model or OpenAI model)
            api_key: OpenAI API key (for OpenAI models only)
            
        Returns:
            Dictionary with coherence scores by player role
        """
        # Available OpenAI embedding models
        valid_openai_models = ["text-embedding-3-large", "text-embedding-ada-002"]
        
        # Получаем API ключ из переменной окружения, если не указан
        if model_type == "openai" and not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(f"Warning: OpenAI embedding requested but OPENAI_API_KEY not found in environment")
                print(f"Falling back to local embedding model")
                model_type = "local"
                model_name = "all-MiniLM-L6-v2"
        
        # Collect CoT data by role
        cot_pairs = []
        player_roles = {}
        
        for result in results:
            spy_id = result.get("spy_id")
            if spy_id is None or not result.get("cot_data", {}):
                continue
                
            for player_id, cot_data in result.get("cot_data", {}).items():
                if not isinstance(cot_data, dict) or "thought" not in cot_data or "speak" not in cot_data:
                    continue
                    
                # Get thought and speech text
                thought = cot_data.get("thought", "")
                speak = cot_data.get("speak", "")
                
                if not thought or not speak:
                    continue
                
                int_player_id = int(player_id)
                cot_pairs.append((thought, speak, int_player_id))
                player_roles[int_player_id] = "spy" if int_player_id == spy_id else "villager"
        
        if not cot_pairs:
            return {"spy": 0.0, "villagers": 0.0, "all": 0.0}
        
        # Calculate embeddings and similarities based on selected model type
        if model_type == "openai" and api_key and model_name in valid_openai_models:
            try:
                import numpy as np
                coherence_scores = []
                
                for thought, speak, player_id in cot_pairs:
                    # Get embeddings for thought using OpenAI API
                    thought_response = requests.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": model_name, "input": thought}
                    )
                    thought_data = thought_response.json()
                    thought_embed = thought_data["data"][0]["embedding"]
                    
                    # Get embeddings for speech using OpenAI API
                    speak_response = requests.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": model_name, "input": speak}
                    )
                    speak_data = speak_response.json()
                    speak_embed = speak_data["data"][0]["embedding"]
                    
                    # Calculate cosine similarity
                    thought_embed_np = np.array(thought_embed)
                    speak_embed_np = np.array(speak_embed)
                    coherence = np.dot(thought_embed_np, speak_embed_np) / (
                        np.linalg.norm(thought_embed_np) * np.linalg.norm(speak_embed_np)
                    )
                    
                    coherence_scores.append((coherence, player_id))
                
                # Calculate average coherence by role
                spy_coherence = [c for c, pid in coherence_scores if player_roles[pid] == "spy"]
                villager_coherence = [c for c, pid in coherence_scores if player_roles[pid] == "villager"]
                
                return {
                    "spy": np.mean(spy_coherence) if spy_coherence else 0.0,
                    "villagers": np.mean(villager_coherence) if villager_coherence else 0.0,
                    "all": np.mean([c for c, _ in coherence_scores]) if coherence_scores else 0.0
                }
                
            except Exception as e:
                print(f"OpenAI API error: {e}")
                print("Falling back to local embedding model")
        
        # Use local sentence transformers model (default or fallback)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            print(f"Using local SentenceTransformer model: {model_name}")
            model = SentenceTransformer(model_name)
            
            coherence_scores = []
            for thought, speak, player_id in cot_pairs:
                # Calculate embeddings
                thought_embed = model.encode(thought)
                speak_embed = model.encode(speak)
                
                # Calculate cosine similarity
                coherence = np.dot(thought_embed, speak_embed) / (
                    np.linalg.norm(thought_embed) * np.linalg.norm(speak_embed)
                )
                
                coherence_scores.append((coherence, player_id))
            
            # Calculate average coherence by role
            spy_coherence = [c for c, pid in coherence_scores if player_roles[pid] == "spy"]
            villager_coherence = [c for c, pid in coherence_scores if player_roles[pid] == "villager"]
            
            return {
                "spy": np.mean(spy_coherence) if spy_coherence else 0.0,
                "villagers": np.mean(villager_coherence) if villager_coherence else 0.0,
                "all": np.mean([c for c, _ in coherence_scores]) if coherence_scores else 0.0
            }
            
        except ImportError:
            print("Warning: sentence-transformers not installed. Using placeholder for CoT coherence.")
            return {"spy": 0.0, "villagers": 0.0, "all": 0.0}
        except Exception as e:
            print(f"Error calculating CoT coherence: {e}")
            return {"spy": 0.0, "villagers": 0.0, "all": 0.0} 