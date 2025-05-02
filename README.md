# PolitAgent Benchmark

PolitAgent Benchmark is a unified framework for evaluating and benchmarking Large Language Models (LLMs) in a variety of multi-agent game environments, including **Spyfall**, **Beast**, **AskGuess**, and **TofuKingdom**.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Supported Games](#supported-games)
- [How It Works](#how-it-works)
- [Running Benchmarks](#running-benchmarks)
- [Analyzing Results](#analyzing-results)
- [Adding a New Game](#adding-a-new-game)
- [License](#license)

## Project Structure

```
PolitAgent-environments/
├── core/
│ ├── benchmark.py # Main benchmarking script
│ ├── benchmark_visualizer.py # Results analysis and visualization
│ └── init.py
├── environments/
│ ├── spyfall/ # Spyfall game environment
│ ├── beast/ # Beast game environment
│ ├── askguess/ # AskGuess game environment
│ └── tofukingdom/ # TofuKingdom game environment
├── llm/ # Unified LLM interface and agent wrappers
├── configs/ # Experiment configuration files (YAML)
├── benchmark_results/ # Output directory for benchmark results
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PolitAgent-environments.git
cd PolitAgent-environments
```

### 2. Install Poetry (if not already installed)

```bash
pip install poetry
```

### 3. Install All Dependencies

```bash
poetry install --no-root
```

This will create a virtual environment and install all required packages, including:
- `langchain`, `langchain-openai`, `mistralai`
- `pandas`, `matplotlib`, `seaborn`
- `func_timeout`
- and others as specified in `pyproject.toml`

### 4. Activate the Poetry Environment

```bash
poetry shell
```

## Supported Games

### 1. Spyfall
A social deduction game where one player is the spy and must guess the secret word, while others try to identify the spy.

### 2. Beast
A game where one model plays as the "beast" and others try to catch it.

### 3. AskGuess
A game where the model must guess a secret word by asking questions.

### 4. TofuKingdom
A three-role game (prince, queen, spy) with asymmetric strategies and hidden information.

## How It Works

- **Benchmarking**: The framework runs multiple games in parallel, assigning LLMs to different roles and collecting results for statistical analysis.
- **Extensibility**: New games can be added as Python packages with minimal changes to the core.
- **LLM Abstraction**: Easily switch between OpenAI, Mistral, or other providers via configuration.
- **Structured Output**: Prompts and outputs are designed for robust parsing and evaluation.

## Running Benchmarks

### Run All Games

```bash
python -m core.benchmark --models openai --workers 4 --runs_per_game 5
```

### Run Specific Games

```bash
python -m core.benchmark --models openai,mistral --games spyfall,askguess --workers 2
```

#### Common Arguments

- `--models`: Comma-separated list of models (e.g., `openai,mistral`)
- `--games`: Comma-separated list of games (`spyfall,beast,askguess,tofukingdom`)
- `--workers`: Number of parallel processes
- `--runs_per_game`: Number of runs per game/phrase
- `--debug`: Enable verbose logging

#### Game-Specific Arguments

**Spyfall:**
- `--label_path`: Path to word pairs file
- `--spy_model_name`: Model for the spy
- `--villager_model_name`: Model for villagers
- `--openai_api_key`: OpenAI API key (overrides env variable)
- `--embedding_model`: Embedding model type (`local`, `openai`, `auto`)
- `--embedding_model_name`: Embedding model name
- `--perplexity_model`: Perplexity model (`auto`, `local`, or model name)

**Beast/AskGuess:**
- `--model_name`: Model name
- `--mode`: Game mode (for AskGuess)

**TofuKingdom:**
- `--prince_model_name`: Model for prince
- `--queen_model_name`: Model for queen
- `--spy_model_name`: Model for spy

## Analyzing Results

After running benchmarks, analyze results with:

```bash
python -m core.benchmark_visualizer --results_dir benchmark_results/20240529_123456
```

- The visualizer will generate plots and a Markdown report in the specified results directory.

## Adding a New Game

1. Create a new directory under `environments/` with your game logic, agents, and prompts.
2. Implement a game class with `init_game` and `game_loop` methods.
3. Register your game in the `GAME_ENVIRONMENTS` dictionary in `core/benchmark.py`.
4. Add any custom metrics to `create_performance_dataframe` in `core/benchmark_visualizer.py`.

## License

MIT

## Contact

For questions or contributions, please open an issue or pull request on GitHub.