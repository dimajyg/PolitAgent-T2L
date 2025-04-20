# Spyfall - AI Agents Social Deduction Game

## Overview
Spyfall is an AI-powered implementation of a social deduction game where multiple AI agents interact with each other. One agent is secretly assigned as a spy, while others are villagers. Each game revolves around a secret word pair, where villagers know one word and the spy knows a different but related word (e.g., "ipad" vs "iphone").

## Game Rules

### Setup
- The game involves 6 AI agents: Nancy, Tom, Cindy, Jack, Rose, and Edward
- One agent is randomly chosen to be the spy
- Each villager receives the same word
- The spy receives a different but related word
- Neither the spy nor the villagers know who has which role

### Gameplay
1. **Describing Stage**
   - Each player takes turns describing their word
   - Villagers try to describe their word without being too obvious
   - The spy tries to blend in without revealing their ignorance of the villagers' word

2. **Voting Stage**
   - After descriptions, players vote on who they think is the spy
   - The player with the most votes is eliminated
   - If the spy is caught, villagers win
   - If the number of living players drops to 3 or fewer, the spy wins

## Project Structure
```
spyfall/
├── game.py              # Main game logic implementation
├── labels.txt           # Word pairs for the game
├── compute_adversarial.py # Adversarial computation utilities
├── agents/              # AI agent implementations
└── utils/              # Utility functions and helpers
```

## Requirements
- Python 3.8+
- Required packages (list them in your environment)
- LLM models for agent behavior

## How to Run

1. **Setup Environment**
```bash
# Create and activate conda environment
conda create -n spyfall python=3.8
conda activate spyfall

# Install required packages
pip install -r requirements.txt

python game_spyfall.py --spy_model_name openai --villager_model_name openai #or change openai on mistral if you use it
```

2. **Run the Game**
```python
from spyfall.game import SpyfallGame

# Initialize game with appropriate models
game = SpyfallGame(args, spy_model, villager_model)

# Start the game with a word pair
game.init_game(("ipad", "iphone"))

# Run the game loop
result = game.game_loop("game.log")
```

## Game Output
The game generates detailed logs including:
- Player descriptions and votes
- Chain of thought reasoning for each agent
- Game outcome (spy or villager victory)
- Number of rounds played

## Word Pairs
The game uses word pairs from `labels.txt`. Each line contains two related words separated by a comma:
- ipad,iphone
- guitar,lute
- BMW,BENZ
etc.

## Customization
You can customize the game by:
- Adding new word pairs to `labels.txt`
- Modifying the number of players
- Adjusting the agent behavior models
- Changing the winning conditions

## Contributing
Feel free to contribute by:
- Adding new word pairs
- Improving agent behavior
- Enhancing game mechanics
- Fixing bugs

## License
MIT