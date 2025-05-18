"""
Унифицированный интерфейс для работы с LLM через LangChain.
Предоставляет единые методы создания моделей для всех игр.
"""

import os
import logging
import importlib
import pkgutil
from typing import Dict, Any, Optional, Union, List, Callable

# LangChain импорты
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
# При необходимости можно добавить другие модели, например:
# from langchain_anthropic import ChatAnthropic
# from langchain_google_vertexai import ChatVertexAI

# Доступные модели
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
    # Можно добавить другие поставщики
}

# Настройки по умолчанию
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
    """Декоратор для регистрации модели в реестре по имени."""
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
    Унифицированная функция для создания LangChain-совместимых LLM-моделей.
    
    Args:
        model_name: Имя провайдера модели ('openai', 'mistral', 'ollama', etc.)
        specific_model: Конкретная модель провайдера (например 'gpt-4' для OpenAI)
        temperature: Температура генерации (0.0-1.0)
        api_key: API ключ (опционально, иначе берется из настроек)
        **kwargs: Дополнительные аргументы для модели
        
    Returns:
        BaseLanguageModel: LangChain-совместимая модель
    """
    logging.info(f"Инициализация модели: {model_name} ({specific_model or 'default'})")
    
    # Получаем настройки по умолчанию для провайдера
    if model_name not in DEFAULT_MODEL_SETTINGS:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступные модели: {list(DEFAULT_MODEL_SETTINGS.keys())}")
    
    defaults = DEFAULT_MODEL_SETTINGS[model_name].copy()
    
    # Переопределяем специфическими настройками
    if specific_model:
        defaults["model"] = specific_model
    if temperature is not None:
        defaults["temperature"] = temperature
    if api_key:
        defaults["api_key"] = api_key
    
    # Объединяем с доп. аргументами
    model_config = {**defaults, **kwargs}
    
    # Создаем модель в зависимости от провайдера
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
        return ChatOllama(
            model=model_config["model"],
            temperature=model_config["temperature"],
            base_url=model_config.get("base_url", "http://localhost:11434"),
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "base_url"]}
        )
    
    # Проверяем, есть ли класс в реестре моделей
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name](**model_config)
    
    # Если ничего не подошло
    raise ValueError(f"Неизвестная модель: {model_name}")

def format_messages(
    system_prompt: str,
    user_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Форматирует сообщения для LLM в нужном формате.
    
    Args:
        system_prompt: Системный промпт
        user_message: Сообщение пользователя (опционально)
        history: История сообщений (опционально)
        
    Returns:
        List[Dict[str, str]]: Список сообщений в формате LangChain
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем историю, если есть
    if history:
        messages.extend(history)
    
    # Добавляем сообщение пользователя, если есть
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    return messages

def get_available_models() -> Dict[str, Callable[..., Any]]:
    """Получить все доступные модели."""
    return _MODEL_REGISTRY.copy()

def get_default_model(model_name: str) -> str:
    """
    Возвращает имя модели по умолчанию для данного провайдера.
    
    Args:
        model_name: Имя провайдера модели
        
    Returns:
        str: Имя модели по умолчанию
    """
    if model_name not in DEFAULT_MODEL_SETTINGS:
        raise ValueError(f"Неизвестный провайдер: {model_name}")
    
    return DEFAULT_MODEL_SETTINGS[model_name]["model"]

# Автоматический импорт всех подмодулей для автодетекта моделей
llm_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([llm_dir]):
    if module_name != "models":
        importlib.import_module(f"llm.{module_name}")

# Пример регистрации vllm-модели (если пакет установлен)
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
    # vLLM через OpenAI-compatible endpoint
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