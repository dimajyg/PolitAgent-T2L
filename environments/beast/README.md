# Beast Game Environment

Beast is a strategic wealth-accumulation game where language model agents negotiate, form alliances, and compete for resources in a multi-round competition.

## Overview

In this game:
- 10 agents compete to accumulate the most wealth through negotiation and strategy
- Players can make wealth transfers to form alliances and strategic partnerships
- Each round ends with a voting phase where one player is eliminated but gains bonus wealth
- The game balances cooperation (forming alliances) with competition (eliminating opponents)
- The goal is to maximize wealth by the end of the game

## Game Rules

1. Each player starts with a random amount of wealth between 0-200,000
2. The game consists of multiple rounds with two phases each:
   - **Negotiation Phase**: Players engage in conversations and can transfer wealth
   - **Voting Phase**: All players vote for one player to eliminate
3. The player with the most votes receives 250,000 wealth but is eliminated from future rounds
4. Only the first 5 eliminated players receive the bonus
5. The game ends after 5 players are eliminated
6. Players win by accumulating the most wealth, not by surviving

## File Structure

```
beast/
├── game.py               # Main game implementation with game loop
├── agents/               # Agent implementations
│   ├── base_agent.py     # BeastAgent class with bargaining and voting logic
│   └── __init__.py       # Package exports
├── utils/                # Utility functions
│   ├── prompt.py         # Prompt template handling
│   └── utils.py          # Helper functions
├── prompts/              # Prompt text files
│   ├── role_prompt.txt               # Basic role description for players
│   ├── choose_conversation_prompt.txt # Instructions for choosing conversation partners
│   ├── conversation_prompt.txt       # Guidelines for negotiation conversations
│   ├── wealth_status_prompt.txt      # Format for displaying current wealth status
│   └── voting_results_prompt.txt     # Format for displaying voting results
└── __init__.py           # Package exports
```

## Running the Game

### As Part of the Benchmark

```bash
python -m core.benchmark --games beast --models openai --runs_per_game 1
```

### With Custom Parameters

```bash
python -m core.benchmark --games beast --models openai --runs_per_game 3 \
    --output_dir ./results/beast_custom \
    --max_rounds 7
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | LLM provider to use | `openai` |
| `--output_dir` | Directory for saving game results | `./results/beast` |
| `--max_rounds` | Number of elimination rounds | `5` |
| `--debug` | Enable verbose logging | `False` |

## Advanced Features

### Structured Output

Beast uses Pydantic models for structured LLM outputs to improve reliability:

- `BargainResponse`: Captures negotiation messages and offers
- `VoteResponse`: Ensures proper vote formatting

The system includes fallback mechanisms to handle potential output parsing errors.

### Game State Tracking

The game saves detailed state information at each round:
- Wealth of all players
- Elimination status
- Voting results
- Conversation history

These files are saved to the specified output directory for later analysis.

## Game Logic

1. **Initialization**:
   - Create 10 players with random initial wealth
   - Initialize agent conversation histories

2. **Conversation Stage**:
   - Update all agents with current wealth status
   - Agents choose opponents to converse with
   - Players engage in conversations and can make wealth transfers
   - Each conversation consists of multiple messages and potential offers

3. **Voting Stage**:
   - Each player votes for another player
   - Player with most votes is eliminated but receives a 250,000 wealth bonus
   - All players are informed about voting results

4. **Game End**:
   - After 5 players are eliminated, the game ends
   - Final rankings are determined by total wealth accumulated
   - Results are saved to JSON for analysis

## Game Results

The game returns a structured result dictionary with:
- `eliminated_players`: List of eliminated players and their final wealth
- `remaining_players`: List of surviving players and their wealth
- `total_rounds`: Number of rounds played
- `game`: Game identifier ("beast")

## Extending the Game

To customize the game behavior:

1. Modify prompt files in the `prompts/` directory to change agent behavior
2. Adjust conversation and voting logic in `game.py`
3. Add new agent strategies by extending the `BeastAgent` class
4. Implement additional bargaining mechanisms or voting strategies 