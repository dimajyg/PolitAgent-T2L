from llm.models import register_model

@register_model("mistral")
class MistralChatModel:
    """
    LangChain-compatible wrapper for Mistral chat models.
    """
    def __init__(self, **kwargs):
        from langchain_mistralai import ChatMistralAI
        self.llm = ChatMistralAI(**kwargs)
    def invoke(self, messages, **kwargs):
        if isinstance(messages, list):
            prompt = "\n".join([m["content"] for m in messages if "content" in m])
        else:
            prompt = str(messages)
        return self.llm.invoke(prompt, **kwargs)
    def with_structured_output(self, schema, **kwargs):
        return self.llm.with_structured_output(schema, **kwargs)