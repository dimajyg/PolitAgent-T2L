from llm.models import register_model

@register_model("ollama")
class OllamaChatModel:
    """
    LangChain-compatible wrapper for Ollama chat models.
    Allows using locally hosted models through Ollama.
    """
    def __init__(self, model="gemma3:latest", base_url="http://localhost:11434", temperature=0.7, **kwargs):
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    
    def invoke(self, messages, **kwargs):
        if isinstance(messages, list):
            # Use proper message formatting for Ollama
            return self.llm.invoke(messages, **kwargs)
        else:
            # Handle string input
            prompt = str(messages)
            return self.llm.invoke(prompt, **kwargs)
    
    def with_structured_output(self, schema, **kwargs):
        return self.llm.with_structured_output(schema, **kwargs) 