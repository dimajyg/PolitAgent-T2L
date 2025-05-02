from chat.mistral_chat import Mistral_Base

def get_model(model_name):
    if model_name == 'mistral':
        return Mistral_Base()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def create_message(role, content):
    return {"role": role, "content": content}