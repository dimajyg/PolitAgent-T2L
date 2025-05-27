
def create_message(role: str, content: str) -> dict:
    """
    Creates a message dictionary in the format expected by LLM API.
    
    Args:
        role (str): The role of the message sender ('system', 'user', 'assistant')
        content (str): The content of the message
        
    Returns:
        dict: Message dictionary
    """
    return {"role": role, "content": content}

def print_messages(messages: list) -> None:
    """Prints a list of messages to the console."""
    for message in messages:
        print(message)

