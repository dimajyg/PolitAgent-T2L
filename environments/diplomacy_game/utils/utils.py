from typing import Dict

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
    Estimates the number of tokens in the text.
    This is a rough estimate, but sufficient for context management.
    
    Args:
        text: Text to estimate
        
    Returns:
        Approximate number of tokens
    """
    return len(text) // 4