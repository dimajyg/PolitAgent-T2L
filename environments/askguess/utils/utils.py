"""
Helper functions for the AskGuess game.
"""
from llm.models import get_model
from typing import Dict, Any, List, Optional

def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Creates a message dictionary in the format expected by LLM API.
    
    Args:
        role (str): Role of the message sender ('system', 'user', 'assistant')
        content (str): Message content
        
    Returns:
        dict: Message dictionary
    """
    return {"role": role, "content": content}

def print_messages(messages: List[Dict[str, str]]) -> None:
    """
    Prints a list of messages to the console.
    
    Args:
        messages: List of messages
    """
    for message in messages:
        print(f"{message['role']}: {message['content']}")

def convert_messages_to_prompt(messages: List[Dict[str, str]], role: str) -> str:
    """
    Converts message history to a string prompt for the model.
    
    Args:
        messages: List of messages in the format [{"role": "...", "content": "..."}]
        role: Agent role ('questioner' or 'answerer')
        
    Returns:
        str: Formatted prompt
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
