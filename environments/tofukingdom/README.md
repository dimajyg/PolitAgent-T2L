# TofuKingdom Game Environment

TofuKingdom is a social deduction game where language models play a medieval court intrigue scenario. The Prince must determine which player is the Princess by asking questions to various characters who may tell the truth or lie depending on their role.

## Overview

In this game:
- 8 different roles interact in a single game session
- The Prince must identify the Princess through strategic questioning
- Different characters have distinct truth-telling behaviors
- Three competing teams vie for victory based on the Prince's final decision
- Complex deduction is required to navigate through deception and misdirection

## Game Rules

1. **Roles and Behaviors**:
   - **Prince**: Must identify the Princess through questioning (always controlled by its own model)
   - **Princess**: Always tells the truth
   - **Chef**: Always tells the truth
   - **Queen**: Always lies
   - **Minister**: Always lies
   - **Guard**: Always lies
   - **Maid**: Can either tell the truth or lie
   - **Spy**: Can either tell the truth or lie

2. **Team Structure**:
   - **Princess Team**: Princess and Chef
   - **Queen Team**: Queen, Minister, and Guard
   - **Neutral Team**: Maid and Spy

3. **Gameplay**:
   - Prince asks a series of questions to all players
   - Questions must be one of three predefined formats:
     1. "Which player is the Princess?"
     2. "What is your identity?"
     3. "What is the identity of {player_name}?"
   - After questioning all players, the Prince can ask one final question
   - The Prince then makes a final guess about who is the Princess

4. **Victory Conditions**:
   - If the Prince correctly identifies the Princess, the Princess Team wins
   - If the Prince wrongly identifies the Queen, the Queen Team wins
   - If the Prince identifies any other character, the Neutral Team wins

## File Structure

```
tofukingdom/
├── game.py               # Main game implementation
├── agents/               # Agent implementations
│   ├── base_agent.py     # Base agent for all roles
│   ├── prince_agent.py   # Prince-specific agent
│   ├── role_agent.py     # Unified agent for all other roles
│   ├── game_controller.py # Game state and management
│   └── __init__.py       # Role and team definitions
├── utils/                # Utility functions
│   ├── prompt.py         # Prompt template handling
│   └── utils.py          # Helper functions
├── prompts/              # Prompt text files
│   ├── game_prompt_en.txt # English game rules
│   ├── game_prompt_zh.txt # Chinese game rules
│   └── role_*.txt        # Role-specific prompts
└── __init__.py           # Package exports
```

## Running the Game

### As Part of the Benchmark

```bash
python -m core.benchmark --games tofukingdom --models openai --runs_per_game 1
```

### With Custom Parameters

```bash
python -m core.benchmark --games tofukingdom --models openai --runs_per_game 3 \
    --prince_model_name openai \
    --princess_model_name openai \
    --queen_model_name mistral \
    --neutral_model_name openai
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prince_model_name` | Model for the Prince agent | Same as general model setting |
| `--princess_model_name` | Model for Princess team (Princess, Chef) | Same as Prince model |
| `--queen_model_name` | Model for Queen team (Queen, Minister, Guard) | Same as Prince model |
| `--neutral_model_name` | Model for Neutral team (Maid, Spy) | Same as Prince model |
| `--n_players` | Number of players (excluding Prince) | `7` |
| `--debug` | Enable verbose logging | `False` |

## Advanced Features

### Role-Based Agent Architecture

TofuKingdom uses a unified agent architecture where:
- All non-Prince roles are handled by the `RoleAgent` class
- Each role's truth behavior is determined by a mapping dictionary
- Team affiliations are tracked for determining victory conditions

### Question System

The Prince agent is restricted to asking three types of questions:
1. Direct question about Princess identity
2. Question about the player's own identity
3. Question about another player's identity

The question system enforces these formats while allowing the Prince to develop strategic questioning approaches.

## Game Logic

1. **Initialization**:
   - Randomly assign roles to players
   - Create role-specific agents with appropriate models
   - Set up the truth-telling behavior for each role

2. **Question Rounds**:
   - Prince asks one question to each player
   - Players respond according to their role's truth behavior
   - Responses are tracked in the Prince's knowledge base

3. **Final Question**:
   - Prince selects one player for a final question
   - The chosen player responds according to their role
   - Prince updates their information one last time

4. **Final Decision**:
   - Prince makes a final guess about who is the Princess
   - The winning team is determined based on the guess
   - Game results are compiled and returned

## Game Results

The game returns a structured result dictionary with:
- `winner_team`: Which team won ("Princess", "Queen", or "Neutral")
- `winners`: List of players on the winning team
- `princess_guess`: The player guessed to be the Princess
- `guessed_role`: The actual role of the guessed player
- `true_princess`: The player who was actually the Princess
- `identities`: Complete mapping of players to roles

## Extending the Game

To customize or extend the game:
1. Modify the prompt files to change agent behavior
2. Adjust the question system in the Prince agent
3. Implement new truth behaviors in the RoleAgent class
4. Add additional roles or teams by updating the role mappings 