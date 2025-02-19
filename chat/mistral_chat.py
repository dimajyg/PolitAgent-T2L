import sys
from mistralai import Mistral
from func_timeout import func_set_timeout
from time import sleep

from chat.config import key_mistral, temperature_mistral, model_mistral

@func_set_timeout(15)
def get_response(messages):
    client = Mistral(api_key=key_mistral)
    response = client.chat.complete(
        model=model_mistral,
        temperature=temperature_mistral,
        messages=messages
    )
    return response

class Mistral_Base:
    def __init__(self) -> None:
        self.name = "mistral"

    def single_chat(self, content, role=None):
        if role is None:
            role = "You are an AI assistant that helps people find information."
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ]
        res = None
        cnt = 0
        while True:
            try:
                response = get_response(messages)
                res = response.choices[0].message.content
                break
            except:
                cnt += 1
            if cnt >= 5:
                break
        return res

    def multi_chat(self, messages):
        res = None
        cnt = 0
        while True:
            sleep(5)
            try:
                response = get_response(messages)
                res = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                cnt += 1
            if cnt >= 3:
                break
        return res