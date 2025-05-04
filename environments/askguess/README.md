# AskGuess Game Environment

AskGuess is a question-answering game where an agent must guess a secret word by asking a series of questions. It simulates a classic word-guessing game where players must use deductive reasoning to identify an unknown object.

## Overview

In this game:
- One agent (QuestionAgent) tries to guess a hidden word
- Another agent (AnswerAgent) provides responses to guide the guessing
- The game has two difficulty modes: "easy" and "hard"
- The goal is to guess the word in as few questions as possible

## Game Modes

### Easy Mode
- The answerer first gives a brief description of the word (without revealing it)
- Then the questioner can ask open-ended questions
- The answerer provides helpful responses and confirms when the word is guessed

### Hard Mode
- No initial description is provided
- The questioner must ask yes/no questions only
- The answerer can only respond with "yes", "no", or "gameover"

## File Structure

```
askguess/
├── game.py               # Main game implementation
├── agents/               # Agent implementations
│   ├── question_agent.py # Agent that asks questions
│   └── answer_agent.py   # Agent that answers questions
├── utils/                # Utility functions
│   ├── prompt.py         # Prompt templates
│   └── utils.py          # Helper functions
├── prompts/              # Prompt text files
│   ├── answerer_easy.txt # Easy mode prompt for answerer
│   ├── answerer_hard.txt # Hard mode prompt for answerer
│   ├── questioner_easy.txt # Easy mode prompt for questioner
│   └── questioner_hard.txt # Hard mode prompt for questioner
├── labels.json           # Full set of words for testing
└── test_labels.json      # Small set of words for quick testing
```

## Running the Game

### As Part of the Benchmark

```bash
python -m core.benchmark --games askguess --models openai --runs_per_game 1
```

### With Custom Parameters

```bash
python -m core.benchmark --games askguess --models openai --runs_per_game 1 \
    --label_path environments/askguess/test_labels.json \
    --mode hard \
    --max_rounds 10
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--label_path` | Path to JSON file with words to guess | `environments/askguess/test_labels.json` |
| `--mode` | Game mode ("easy" or "hard") | `hard` |
| `--max_rounds` | Maximum number of question rounds | `10` |
| `--model_name` | LLM provider to use | `openai` |
| `--max_phrases` | Limit number of words to test | All words in the file |

## Game Logic

1. **Initialization**:
   - Load a word from the labels file
   - Set up QuestionAgent and AnswerAgent with the chosen model
   - Initialize game state and history

2. **In Easy Mode**:
   - Start with a description phase where the answerer provides clues
   - Check that the description doesn't mention the word directly

3. **Q&A Rounds**:
   - The questioner asks a single question per round
   - The answerer provides a response
   - Responses are checked to ensure they don't reveal the word

4. **Game End Conditions**:
   - The word is correctly guessed (success)
   - Maximum rounds are reached (partial success)
   - An error occurs (failure)
   - The answerer mentions the word directly (cheating failure)

## Game Results

The game returns a result dictionary with:
- `object`: The word that was being guessed
- `round`: The round in which the game ended (-1 for errors)
- `qa_history`: Complete history of questions and answers
- `error_type`: Type of outcome (success, failure, etc.)

## Label Format

The label file should be a JSON array of strings:

```json
["apple", "computer", "elephant", "umbrella"]
```

## Adding New Words

To add new words to guess:
1. Edit `labels.json` to include additional words
2. For testing, you can create a custom labels file and specify it with `--label_path`

## Extending the Game

To modify or extend the game:
1. Customize prompts in the `prompts/` directory
2. Add new features to agent implementations
3. Modify game logic in `game.py` 