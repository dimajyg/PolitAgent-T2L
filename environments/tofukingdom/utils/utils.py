from chat.mistral_chat import Mistral_Base

def get_model(model_name):
    return Mistral_Base()

def create_message(role,content):
    return {"role":role,"content":content}

def print_messages(messages):
    for message in messages:
        print(message)

