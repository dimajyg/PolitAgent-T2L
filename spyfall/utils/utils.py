from chat.mistral_chat import Mistral_Base
from chat.openai_chat import OpenAI_Base

def get_model(model_name):
    if model_name == "mistral":
        return Mistral_Base()
    elif model_name == "openai":
        return OpenAI_Base()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def create_message(role,content):
    return {"role":role,"content":content}

def print_messages(messages):
    for message in messages:
        print(message)

