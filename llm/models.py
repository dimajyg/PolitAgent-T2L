import os
import logging
import importlib
import pkgutil
from typing import Dict, Any, Optional, Union, List, Callable

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

AVAILABLE_MODELS = {
    "openai": {
        "gpt-3.5-turbo": {"max_tokens": 4096, "description": "Быстрая и экономичная модель"},
        "gpt-4-turbo": {"max_tokens": 8192, "description": "Продвинутая модель с улучшенным пониманием"},
    },
    "mistral": {
        "mistral-tiny": {"max_tokens": 4096, "description": "Маленькая быстрая модель"},
        "mistral-small": {"max_tokens": 8192, "description": "Сбалансированная модель"},
        "mistral-medium": {"max_tokens": 8192, "description": "Продвинутая модель"},
    },
    "ollama": {
        "llama2": {"max_tokens": 4096, "description": "Локальная модель Llama 2"},
        "mistral": {"max_tokens": 4096, "description": "Локальная модель Mistral"},
        "phi2": {"max_tokens": 2048, "description": "Легкая и быстрая модель"},
        "gemma": {"max_tokens": 4096, "description": "Модель Gemma от Google"},
    },
}

DEFAULT_MODEL_SETTINGS = {
    "openai": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "api_key": os.environ.get("OPENAI_API_KEY", None),
    },
    "mistral": {
        "model": "mistral-small",
        "temperature": 0.7,
        "api_key": os.environ.get("MISTRAL_API_KEY", None),
    },
    "ollama": {
        "model": "llama2",
        "temperature": 0.7,
        "base_url": "http://localhost:11434",
    }
}

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    """Decorator for registering a model in the registry by name."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(
    model_name: str, 
    specific_model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    **kwargs: Any
) -> BaseLanguageModel:
    """
    Unified function for creating LangChain-compatible LLM models.
    
    Args:
        model_name: Name of the model provider ('openai', 'mistral', 'ollama', etc.)
        specific_model: Specific model from the provider (e.g. 'gpt-4' for OpenAI)
        temperature: Generation temperature (0.0-1.0)
        api_key: API key (optional, otherwise taken from settings)
        **kwargs: Additional model arguments
        
    Returns:
        BaseLanguageModel: LangChain-compatible model
    """
    logging.info(f"Initializing model: {model_name} ({specific_model or 'default'})")
    
    if model_name not in DEFAULT_MODEL_SETTINGS:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступные модели: {list(DEFAULT_MODEL_SETTINGS.keys())}")
    
    defaults = DEFAULT_MODEL_SETTINGS[model_name].copy()
    
    if specific_model:
        defaults["model"] = specific_model
    if temperature is not None:
        defaults["temperature"] = temperature
    if api_key:
        defaults["api_key"] = api_key
    
    model_config = {**defaults, **kwargs}
    
    if model_name == "openai":
        return ChatOpenAI(
            model_name=model_config["model"],
            temperature=model_config["temperature"],
            openai_api_key=model_config["api_key"],
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "api_key"]}
        )
    
    elif model_name == "mistral":
        return ChatMistralAI(
            model=model_config["model"],
            temperature=model_config["temperature"],
            mistral_api_key=model_config["api_key"],
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "api_key"]}
        )
    
    elif model_name == "ollama":
        from langchain_community.chat_models import ChatOllama
        import requests
        from requests.exceptions import RequestException
        
        base_url = model_config.get("base_url", "http://localhost:11434")
        try:
            requests.get(f"{base_url}/api/tags", timeout=3)
            logging.info(f"Successfully connected to Ollama at {base_url}")
        except RequestException as e:
            logging.error(f"Could not connect to Ollama at {base_url}: {e}")
            logging.error("Make sure Ollama is running and accessible")
            raise ConnectionError(f"Could not connect to Ollama at {base_url}. Make sure the Ollama server is running.")
            
        return ChatOllama(
            model=model_config["model"],
            temperature=model_config["temperature"],
            base_url=base_url,
            request_timeout=60.0,
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "base_url"]}
        )
    
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name](**model_config)
    
    raise ValueError(f"Неизвестная модель: {model_name}")

def format_messages(
    system_prompt: str,
    user_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Formats messages for LLM in the correct format.
    
    Args:
        system_prompt: System prompt
        user_message: User message (optional)
        history: Message history (optional)
        
    Returns:
        List[Dict[str, str]]: List of messages in LangChain format
    """
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)
    
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    return messages

def get_available_models() -> Dict[str, Callable[..., Any]]:
    """Get all available models."""
    return _MODEL_REGISTRY.copy()

def get_default_model(model_name: str) -> str:
    """
    Returns the default model name for a given provider.
    
    Args:
        model_name: Name of the model provider
        
    Returns:
        str: Default model name
    """
    if model_name not in DEFAULT_MODEL_SETTINGS:
        raise ValueError(f"Unknown provider: {model_name}")
    
    return DEFAULT_MODEL_SETTINGS[model_name]["model"]

# Automatic import of all submodules for model detection
llm_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([llm_dir]):
    if module_name != "models":
        importlib.import_module(f"llm.{module_name}")

@register_model("any_ollama")
class AnyOllamaChat:
    """
    A robust implementation of Ollama chat model with better error handling.
    Falls back gracefully when Ollama is not available.
    """
    def __init__(self, model="llama2", temperature=0.7, base_url="http://localhost:11434", 
                 timeout=30, **kwargs):
        self.model_name = model
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout
        self.kwargs = kwargs
        self.is_ollama_available = self._check_ollama_availability()
        
        if self.is_ollama_available:
            try:
                from langchain_community.chat_models import ChatOllama
                self.chat_model = ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=base_url,
                    request_timeout=timeout,
                    **kwargs
                )
                logging.info(f"Successfully initialized Ollama model: {model}")
            except Exception as e:
                logging.error(f"Failed to initialize Ollama model: {e}")
                self.is_ollama_available = False
    
    def _check_ollama_availability(self):
        """Check if Ollama server is available."""
        import requests
        from requests.exceptions import RequestException
        
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except RequestException:
            logging.warning(f"Ollama server not available at {self.base_url}")
            return False
    
    def invoke(self, messages, **kwargs):
        """Invoke the model with fallback mechanisms."""
        if not self.is_ollama_available:
            return self._generate_fallback_response(messages)
            
        try:
            result = self.chat_model.invoke(messages, **kwargs)
            return result
        except Exception as e:
            logging.error(f"Error invoking Ollama model: {e}")
            return self._generate_fallback_response(messages)
    
    def _generate_fallback_response(self, messages):
        """Generate a fallback response when Ollama is unavailable."""
        from langchain_core.messages import AIMessage
        
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role", "") == "user":
                user_message = msg.get("content", "")
                break
        
        if "describe" in user_message.lower():
            content = "I'm not sure how to describe this properly."
            if "spy" in user_message.lower():
                content = '{"thought": "I need to be vague but not obvious", "speak": "It\'s something you might encounter in daily life."}'
            else:
                content = '{"thought": "I need to be specific but not too obvious", "speak": "It\'s something we use regularly."}'
        elif "vote" in user_message.lower():
            # Try to extract living players from the message
            import re
            import random
            import json
            
            players = []
            players_match = re.search(r'Living players: (\[.*?\])', user_message)
            if players_match:
                try:
                    players = json.loads(players_match.group(1))
                except:
                    pass
            
            if players:
                target = random.choice(players)
                content = f'{{"thought": "Making a random choice due to model limitations", "speak": "I think {target} is acting suspicious", "name": "{target}"}}'
            else:
                content = '{"thought": "No information about players", "speak": "I\'m not sure who to vote for", "name": ""}'
        else:
            content = "Sorry, I can't process that request right now."
            
        return AIMessage(content=content)

try:
    from langchain_community.llms import VLLM
    @register_model("vllm")
    class VLLMChatModel:
        def __init__(self, model="mosaicml/mpt-7b", trust_remote_code=True, max_new_tokens=128, top_k=10, top_p=0.95, temperature=0.8, tensor_parallel_size=1, vllm_kwargs=None, **kwargs):
            self.llm = VLLM(
                model=model,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                vllm_kwargs=vllm_kwargs or {},
            )
        def invoke(self, messages, **kwargs):
            if isinstance(messages, list):
                prompt = "\n".join([m["content"] for m in messages if "content" in m])
            else:
                prompt = str(messages)
            return self.llm.invoke(prompt, **kwargs)
        def with_structured_output(self, schema, **kwargs):
            return self.llm.with_structured_output(schema, **kwargs)
    from langchain_community.llms import VLLMOpenAI
    @register_model("vllm_server")
    class VLLMOpenAIChatModel:
        def __init__(self, openai_api_base="http://localhost:8000/v1", model_name="tiiuae/falcon-7b", openai_api_key="EMPTY", model_kwargs=None, **kwargs):
            self.llm = VLLMOpenAI(
                openai_api_base=openai_api_base,
                model_name=model_name,
                openai_api_key=openai_api_key,
                model_kwargs=model_kwargs or {},
                **kwargs
            )
        def invoke(self, messages, **kwargs):
            if isinstance(messages, list):
                prompt = "\n".join([m["content"] for m in messages if "content" in m])
            else:
                prompt = str(messages)
            return self.llm.invoke(prompt, **kwargs)
        def with_structured_output(self, schema, **kwargs):
            return self.llm.with_structured_output(schema, **kwargs)
except ImportError:
    pass 