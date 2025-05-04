"""
Унифицированный интерфейс для работы с LLM через LangChain.
Предоставляет единые методы создания моделей для всех игр.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List

# LangChain импорты
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import Runnable
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
    }
}

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
        model_name: Имя провайдера модели ('openai', 'mistral', etc.)
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
    
    # Можно добавить другие модели по аналогии
    else:
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

def get_available_models() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Возвращает информацию о доступных моделях.
    
    Returns:
        Dict: Словарь с информацией о моделях
    """
    return AVAILABLE_MODELS

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