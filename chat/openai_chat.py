from openai import OpenAI
from func_timeout import func_set_timeout
from time import sleep

from chat.config import openai_api_key, temperature_openai, model_openai

@func_set_timeout(15)
def get_response(messages):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model_openai,
        temperature=temperature_openai,
        messages=messages
    )
    return response

class OpenAI_Base:
    def __init__(self) -> None:
        self.name = "openai"

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