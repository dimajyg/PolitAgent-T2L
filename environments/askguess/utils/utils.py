from chat.mistral_chat import Mistral_Base

def get_model(model_name):
    return Mistral_Base()

def create_message(role,content):
    return {"role":role,"content":content}

def print_messages(messages):
    for message in messages:
        print(message)

def convert_messages_to_prompt(messages,role):
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
