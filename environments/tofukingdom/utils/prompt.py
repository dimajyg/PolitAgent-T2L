"""
Module for loading and providing prompts for the TofuKingdom game using LangChain PromptTemplate.
"""
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, Optional, Union
import logging

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Create prompts directory if it doesn't exist
PROMPT_DIR.mkdir(exist_ok=True, parents=True)

def load_prompt(filename: str) -> str:
    """Loads a prompt from a text file.

    Args:
        filename (str): Name of the file containing the prompt.

    Returns:
        str: The prompt content.
    """
    prompt_path = PROMPT_DIR / filename
    
    # Check if file exists
    if not prompt_path.exists():
        logging.error(f"Prompt file '{filename}' not found in {PROMPT_DIR}")
        raise FileNotFoundError(f"Prompt file '{filename}' not found in {PROMPT_DIR}")
    
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()

def get_game_prompt_template(language: str = "en") -> PromptTemplate:
    """Returns the PromptTemplate for the game rules.

    Args:
        language (str): Language code ("en" for English, "zh" for Chinese)

    Returns:
        PromptTemplate: The game rules prompt template.
    """
    # Default to English if unsupported language is requested
    if language not in ["en", "zh"]:
        logging.warning(f"Unsupported language '{language}'. Defaulting to English.")
        language = "en"
        
    filename = f"game_prompt_{language}.txt"
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

def get_role_prompt_template(role: str) -> PromptTemplate:
    """Returns the PromptTemplate for a specific role.

    Args:
        role (str): The role name (Prince, Princess, etc.)

    Returns:
        PromptTemplate: The role prompt template.
    """
    role_file = f"role_{role.lower()}.txt"
    template = load_prompt(role_file)
    return PromptTemplate.from_template(template)

def format_prompt(prompt_template: Union[PromptTemplate, str], **kwargs: Any) -> str:
    """Format a prompt template with the given variables.
    
    Args:
        prompt_template: Either a PromptTemplate or a template filename
        **kwargs: Variables to format the template with
        
    Returns:
        str: The formatted prompt
    """
    # Default variables for common placeholders
    default_vars = {
        'player_name': 'Player',  # Default player name
        'opponent_name': 'Opponent',  # Default opponent name
        'role_name': 'Unknown',  # Default role name
    }
    
    # Combine default vars with provided kwargs (kwargs take precedence)
    format_vars = {**default_vars, **kwargs}
    
    if isinstance(prompt_template, str):
        template = load_prompt(prompt_template)
        prompt_template = PromptTemplate.from_template(template)
    
    try:
        return prompt_template.format(**format_vars)
    except KeyError as e:
        logging.warning(f"Missing variable in prompt template: {e}")
        # Return template with unfilled variables for debugging
        return f"[ERROR: Missing variable {e} in template] {prompt_template.template}"

# For backwards compatibility - provide game prompts as strings
def get_game_prompt_en() -> str:
    """Returns the English game prompt as a string.
    
    Returns:
        str: The English game prompt
    """
    return load_prompt("game_prompt_en.txt")