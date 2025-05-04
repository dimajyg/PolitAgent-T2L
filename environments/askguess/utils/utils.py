"""
Вспомогательные функции для игры AskGuess.
"""
from llm.models import get_model
from typing import Dict, Any, List, Optional

def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Создает словарь сообщения в формате, ожидаемом LLM API.
    
    Args:
        role (str): Роль отправителя сообщения ('system', 'user', 'assistant')
        content (str): Содержимое сообщения
        
    Returns:
        dict: Словарь сообщения
    """
    return {"role": role, "content": content}

def print_messages(messages: List[Dict[str, str]]) -> None:
    """
    Печатает список сообщений в консоль.
    
    Args:
        messages: Список сообщений
    """
    for message in messages:
        print(f"{message['role']}: {message['content']}")

def convert_messages_to_prompt(messages: List[Dict[str, str]], role: str) -> str:
    """
    Конвертирует историю сообщений в строковый промпт для модели.
    
    Args:
        messages: Список сообщений в формате [{"role": "...", "content": "..."}]
        role: Роль агента ('questioner' или 'answerer')
        
    Returns:
        str: Форматированный промпт
    """
    prompt = ""
    if role == "questioner":
        for message in messages:
            content = message["content"]
            if message["role"] == "user":
                prompt += f"questioner: {content}\n"
            elif message["role"] == "assistant":
                prompt += f"answerer: {content}\n"
            else:
                prompt += f"host: {content}\n"
        prompt += "questioner: "
    else:
        for message in messages:
            content = message["content"]
            if message["role"] == "assistant":
                prompt += f"questioner: {content}\n"
            elif message["role"] == "user":
                prompt += f"answerer: {content}\n"
            else:
                prompt += f"host: {content}\n"
        prompt += "answerer: "
        
    return prompt
