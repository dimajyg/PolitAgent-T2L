"""
Вспомогательные функции для игры Spyfall.
"""
# Импортируем новый унифицированный интерфейс
from llm.models import get_model, format_messages
from environments.spyfall.utils.prompt import game_prompt_en

def create_message(role, content):
    """
    Создает словарь сообщения в формате, ожидаемом LLM API.
    
    Args:
        role (str): Роль отправителя сообщения ('system', 'user', 'assistant')
        content (str): Содержимое сообщения
        
    Returns:
        dict: Словарь сообщения
    """
    return {"role": role, "content": content}

def print_messages(messages):
    """Печатает список сообщений в консоль."""
    for message in messages:
        print(message)

