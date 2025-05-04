# Spyfall Game Environment

Spyfall is a social deduction game where language models play the roles of agents trying to identify the spy among them. It simulates a scenario where one player has different information than the rest of the group.

## Overview

In this game:
- 6 agents participate, with one randomly assigned as the spy
- Each villager receives the same word, while the spy receives a related but different word
- Players must describe their word without directly revealing it
- Through careful observation and deduction, players vote to eliminate the suspected spy
- Either the spy successfully blends in and wins, or the villagers correctly identify the spy

## Game Rules

1. **Setup**:
   - 6 players are assigned with names: Nancy, Tom, Cindy, Jack, Rose, and Edward
   - One player is randomly chosen as the spy
   - Villagers receive a common word (e.g., "planet")
   - The spy receives a related but different word (e.g., "moon")

2. **Gameplay**:
   - The game proceeds in rounds with two phases: description and voting
   - In each round, players must describe their word without directly saying it
   - After all descriptions, players vote on who they think is the spy
   - The player with the most votes is eliminated

3. **Victory Conditions**:
   - If the spy is eliminated, the villagers win
   - If the number of players drops below 3 and the spy remains, the spy wins

## File Structure

```
spyfall/
├── game.py               # Main game implementation
├── agents/               # Agent implementations
│   └── base_agent.py     # BaseAgent class implementing spy and villager behaviors
├── utils/                # Utility functions
│   ├── prompt.py         # Prompt template handling
│   └── utils.py          # Helper functions
├── prompts/              # Prompt text files
│   ├── game_prompt.txt   # Basic game rules description
│   ├── villager_prompt.txt # Specific instructions for villager agents
│   └── spy_prompt.txt    # Specific instructions for the spy agent
├── compute_adversarial.py # Tools for adversarial analysis
├── spyfall_metrics.py    # Metrics for evaluating game performance
└── labels.txt            # Word pairs used in the game
```

## Running the Game

### As Part of the Benchmark

```bash
python -m core.benchmark --games spyfall --models openai --runs_per_game 1
```

### With Custom Parameters

```bash
python -m core.benchmark --games spyfall --models openai,mistral --runs_per_game 3 \
    --label_path environments/spyfall/prompts/labels.txt \
    --spy_model_name openai \
    --villager_model_name mistral
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--label_path` | Path to word pairs file | `environments/spyfall/prompts/labels.txt` |
| `--spy_model_name` | Model for the spy agent | Same as general model setting |
| `--villager_model_name` | Model for villager agents | Same as general model setting |
| `--embedding_model` | Embedding model for analysis | `auto` |
| `--embedding_model_name` | Name of embedding model | `text-embedding-3-large` |
| `--perplexity_model` | Model for perplexity calculation | `auto` |
| `--n` | Number of word pairs to process | `10` |
| `--max_phrases` | Limit number of word pairs to test | All pairs in the file |

## Advanced Features

### Performance Metrics

The Spyfall environment includes sophisticated metrics for analyzing agent behavior:

- **Description Specificity**: Measures how specific or vague a player's description is
- **Perplexity Analysis**: Evaluates the linguistic consistency of descriptions
- **Vagueness Scores**: Quantifies the ambiguity in player descriptions
- **Voting Patterns**: Tracks which players vote for whom and why

### Chain of Thought Reasoning

Agents capture their internal reasoning process as they:
- Analyze other players' descriptions
- Decide how to describe their own word
- Determine who to vote for as the spy

## Game Logic

1. **Initialization**:
   - Load a word pair from the labels file
   - Randomly assign spy and villager roles
   - Initialize agents with appropriate models and roles

2. **Describing Stage**:
   - Each player takes a turn describing their assigned word
   - Players must be careful to be neither too specific nor too vague
   - Descriptions are recorded and analyzed

3. **Voting Stage**:
   - Each player votes for who they think is the spy
   - Players explain their reasoning for their vote
   - The player with the most votes is eliminated
   - If the spy is eliminated, villagers win

4. **Game End Conditions**:
   - The spy is eliminated (villager victory)
   - Less than 3 players remain with the spy still active (spy victory)

## Game Results

The game returns a comprehensive result dictionary with:
- `winner`: Which side won ("spy" or "villager")
- `players`: List of all player names
- `spy_index`: Position of the spy in the player list
- `spy_caught`: Whether the spy was successfully identified
- `votes`: Voting records from each player
- `vote_sequence`: Order and targets of votes
- `descriptions`: Each player's description of their word
- `description_metrics`: Analytical metrics about descriptions
- `vagueness_scores`: Quantitative measure of description vagueness

## Word Pair Format

The label file should contain comma-separated word pairs, one pair per line:

```
planet,moon
iphone,ipad
car,bicycle
```

## Extending the Game

To customize or extend the game:
1. Add new word pairs to the labels.txt file
2. Modify the prompt templates to change agent behavior
3. Adjust the number of players or game mechanics in game.py
4. Implement new metrics in spyfall_metrics.py