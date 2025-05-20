from typing import Dict, Any

def create_message(role: str, content: str) -> Dict[str, str]:
    """Create a message in the format expected by the LLM models.
    
    Args:
        role: The role of the sender (system, user, or assistant)
        content: The content of the message
        
    Returns:
        Dict with role and content
    """
    return {"role": role, "content": content}