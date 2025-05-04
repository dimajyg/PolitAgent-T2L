# Beast Game Environment

Beast is a strategic wealth-accumulation game where language model agents negotiate, form alliances, and compete for resources.

## Game Description

In Beast, players engage in a strategic wealth game with the following rules:

1. Players start with a random amount of initial wealth.
2. Each round consists of conversational negotiations and a voting phase.
3. During conversations, players can make wealth transfer offers to each other.
4. At the end of each round, all players vote for one player.
5. The player with the most votes receives 250,000 wealth but is eliminated from future rounds.
6. Only the first 5 eliminated players receive this bonus.
7. The game continues until 5 players are eliminated.
8. The goal is to accumulate the most wealth by the end of the game.

## Code Structure

- `agents/`: Contains the agent implementations
  - `base_agent.py`: Defines the BeastAgent class using LangChain
  - `__init__.py`: Package exports
- `prompts/`: Contains prompt templates
  - `role_prompt.txt`: Basic role description for players
  - `choose_conversation_prompt.txt`: Instructions for choosing conversation partners
  - `conversation_prompt.txt`: Guidelines for negotiation conversations
  - `wealth_status_prompt.txt`: Format for displaying current wealth status
  - `voting_results_prompt.txt`: Format for displaying voting results
- `utils/`: Utility functions
  - `prompt.py`: Prompt handling using LangChain PromptTemplate
  - `utils.py`: General utility functions
- `game.py`: Main game loop and logic
- `__init__.py`: Package exports

## Agent Architecture

The Beast environment uses the unified language model interface with LangChain integration:

1. Each agent is an instance of `BeastAgent`, inheriting from the central `BaseAgent` architecture.
2. Agents use LangChain's prompt templates for structured LLM interactions.
3. All communication happens through standardized message formats.
4. The codebase follows modern Python best practices:
   - Comprehensive type annotations
   - Proper error handling
   - Detailed docstrings
   - Clear function and variable names

## Key Features

- **Modern LangChain Integration**: Uses the latest LangChain pipeline approach (`prompt | llm`) instead of deprecated LLMChain.
- **Prompt Management**: Prompts are stored in separate text files, loaded dynamically, and formatted using LangChain's PromptTemplate.
- **Robust Parsing**: JSON responses are parsed with intelligent error handling and markdown cleanup.
- **State Management**: Game state is saved at each round for analysis and debugging.
- **Error Tolerance**: The system is designed to gracefully handle model output errors.

## Usage

The Beast game can be run through the benchmark interface:

```bash
python -m core.benchmark --games=beast --model_name=mistral
```

For custom configurations:

```bash
python -m core.benchmark --games=beast --model_name=openai --runs_per_game=3
```

## Extending the Game

To customize the game behavior:

1. Modify prompt files in the `prompts/` directory to change agent behavior
2. Adjust conversation and voting logic in `game.py`
3. Add new agent strategies by extending the `BeastAgent` class 