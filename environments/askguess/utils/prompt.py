"""
Модуль для загрузки и предоставления промптов игры AskGuess через LangChain PromptTemplate.
"""
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from typing import Any

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
    """Загружает промпт из текстового файла.

    Args:
        filename (str): Имя файла с промптом.

    Returns:
        str: Содержимое промпта.
    """
    with open(PROMPT_DIR / filename, encoding="utf-8") as f:
        return f.read()

# Примеры функций для получения PromptTemplate

def get_answerer_prompt_template(mode: str) -> PromptTemplate:
    """Возвращает PromptTemplate для роли answerer.

    Args:
        mode (str): Режим игры ('easy' или 'hard').

    Returns:
        PromptTemplate: Шаблон промпта для answerer.
    """
    filename = f"answerer_{mode}.txt"
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

def get_questioner_prompt_template(mode: str) -> PromptTemplate:
    """Возвращает PromptTemplate для роли questioner.

    Args:
        mode (str): Режим игры ('easy' или 'hard').

    Returns:
        PromptTemplate: Шаблон промпта для questioner.
    """
    filename = f"questioner_{mode}.txt"
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

def get_host_description_prompt() -> str:
    """Возвращает промпт для описания слова ведущим.

    Returns:
        str: Промпт для описания слова.
    """
    return load_prompt("host_description.txt")

def get_host_qa_prompt() -> str:
    """Возвращает промпт для начала Q&A раунда.

    Returns:
        str: Промпт для Q&A раунда.
    """
    return load_prompt("host_qa.txt")