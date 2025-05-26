from typing import Dict, List, Any
import json

def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Creates a message dictionary in the format required by language models.
    
    Args:
        role: The role of the message sender (e.g., 'system', 'user', 'assistant')
        content: The content of the message
        
    Returns:
        Dict with role and content
    """
    return {"role": role, "content": content}

def estimate_tokens(text: str) -> int:
    """
    Примерно оценивает количество токенов в тексте.
    Это грубая оценка, но достаточная для управления контекстом.
    
    Args:
        text: Текст для оценки
        
    Returns:
        Приблизительное количество токенов
    """
    # Примерно 4 символа на токен для английского языка
    return len(text) // 4