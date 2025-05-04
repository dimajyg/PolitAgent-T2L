"""
Utility functions for the TofuKingdom game environment.
"""
from typing import Dict, Any, List, Optional

def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Create a message dictionary for LLM chat history.
    
    Args:
        role: Message role ('system', 'user', or 'assistant')
        content: Message content
        
    Returns:
        Dict with role and content keys
    """
    return {"role": role, "content": content}

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history for display or logging.
    
    Args:
        history: List of message dictionaries
        
    Returns:
        Formatted string representation of the conversation
    """
    result = ""
    for message in history:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            result += f"SYSTEM: {content}\n\n"
        elif role == "user":
            result += f"USER: {content}\n\n"
        elif role == "assistant":
            result += f"ASSISTANT: {content}\n\n"
    
    return result

def print_messages(messages: List[Dict[str, str]]) -> None:
    """
    Print messages to console for debugging.
    
    Args:
        messages: List of message dictionaries
    """
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"[{role.upper()}] {content}")
        print("-" * 80)

