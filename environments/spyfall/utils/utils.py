from llm.mistral_chat import MistralChat
from llm.openai_chat import OpenAIChat
from environments.spyfall.utils.prompt import game_prompt_en

def get_model(model_name):
    """
    Создает и возвращает объект чата на основе имени модели.
    
    Args:
        model_name (str): Название модели - 'mistral' или 'openai'
        
    Returns:
        BaseChat: Объект чата для взаимодействия с LLM
    """
    if model_name == "mistral":
        return MistralChat(system_prompt=game_prompt_en, model_name="mistral-medium")
    elif model_name == "openai":
        return OpenAIChat(system_prompt=game_prompt_en, model_name="gpt-3.5-turbo")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

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

