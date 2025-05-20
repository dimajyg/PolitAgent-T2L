# PolitAgent Benchmark

PolitAgent Benchmark is a unified framework for evaluating and benchmarking Large Language Models (LLMs) in a variety of multi-agent game environments, including **Spyfall**, **Beast**, **AskGuess**, and **TofuKingdom**.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Supported Models](#supported-models)
- [Supported Games](#supported-games)
- [How It Works](#how-it-works)
- [Running Benchmarks](#running-benchmarks)
- [Analyzing Results](#analyzing-results)
- [Adding a New Game](#adding-a-new-game)
- [Adding a New Model](#adding-a-new-model)
- [Using Ollama Models](#using-ollama-models)
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
│ ├── models.py # Model registry and unified interface
│ ├── base_chat.py # Base chat interface
│ ├── openai_chat.py # OpenAI implementation
│ ├── mistral_chat.py # Mistral implementation
│ └── agent.py # Agent abstractions
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

## Supported Models

PolitAgent uses a flexible model registry system to support multiple LLM providers:

- **OpenAI**: GPT-3.5-Turbo, GPT-4, etc.
- **Mistral AI**: Mistral-tiny, Mistral-small, Mistral-medium
- **Ollama**: Locally hosted models like Llama 2, Mistral, Phi-2, Gemma, etc.
- **vLLM**: Both direct and OpenAI-compatible endpoint support
- **Custom Models**: Easily add new models by implementing the base interface

Each model provider can be configured with default settings or customized per experiment.

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

- **Model Registry**: Plugins register with a decorator system, making it easy to add new model providers.
- **Benchmarking**: The framework runs multiple games in parallel, assigning LLMs to different roles and collecting results.
- **Model Abstraction**: A unified interface allows seamless switching between OpenAI, Mistral, vLLM, or custom providers.
- **Structured Output**: Support for Pydantic schemas to ensure consistent model responses.
- **Fallback Mechanisms**: Graceful handling of model failures with fallback strategies.

## Running Benchmarks

### Run All Games

```bash
python -m core.benchmark --models openai --workers 4 --runs_per_game 5
```

### Run Specific Games

```bash
python -m core.benchmark --models openai,mistral --games spyfall,askguess --workers 2
```

### Run with Locally Hosted Models via Ollama

```bash
# Make sure Ollama is running locally
python -m core.benchmark --models ollama --specific_model llama2 --games beast --workers 1
```

#### Common Arguments

- `--models`: Comma-separated list of models (e.g., `openai,mistral,vllm`)
- `--games`: Comma-separated list of games (`spyfall,beast,askguess,tofukingdom`)
- `--workers`: Number of parallel processes
- `--runs_per_game`: Number of runs per game/phrase
- `--debug`: Enable verbose logging
- `--max_phrases`: Limit number of phrases/labels to process (useful for testing)

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
- `--princess_model_name`: Model for princess
- `--queen_model_name`: Model for queen
- `--neutral_model_name`: Model for neutral character

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

## Adding a New Model

1. Create a new module in the `llm/` directory.
2. Implement a class that follows the interface pattern of existing models.
3. Use the `@register_model` decorator to register your model:

```python
from llm.models import register_model

@register_model("your_model_name")
class YourModelClass:
    def __init__(self, **kwargs):
        # Initialize your model
        pass
        
    def invoke(self, messages, **kwargs):
        # Implement the invocation logic
        pass
        
    def with_structured_output(self, schema, **kwargs):
        # Implement structured output support
        pass
```

The model will automatically be discovered and available in benchmarks.

## Using Ollama Models

To use locally hosted models via [Ollama](https://ollama.ai/), follow these steps:

1. **Install Ollama** from [https://ollama.ai/](https://ollama.ai/)

2. **Pull the models** you want to use:
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull phi2
   ollama pull gemma
   ```

3. **Run the benchmark** with Ollama models:
   ```bash
   # Basic usage with default settings
   python -m core.benchmark --models ollama --games beast --workers 1
   
   # Specify a particular model
   python -m core.benchmark --models ollama --specific_model mistral --games beast --workers 1
   
   # Custom Ollama server URL (if not using default localhost)
   python -m core.benchmark --models ollama --specific_model llama2 --ollama_base_url "http://your-server:11434" --games beast --workers 1
   
   # Mixed model usage (OpenAI for spy, Ollama for villagers)
   python -m core.benchmark --games spyfall --spy_model_name openai --villager_model_name ollama --specific_model mistral --workers 1
   ```

4. **Available Ollama models**:
   - `llama2` - Meta's Llama 2 model
   - `mistral` - Mistral AI's model
   - `phi2` - Microsoft's Phi-2 model
   - `gemma` - Google's Gemma model
   - Any other model you've pulled to your Ollama server

### Note on Performance

Local models through Ollama may have different performance characteristics compared to cloud models. Consider these tips:
- Start with smaller games like Beast or AskGuess
- Local models typically work better with fewer agents
- Adjust temperature settings if responses seem too random or too deterministic

## License

MIT

## Contact

For questions or contributions, please open an issue or pull request on GitHub.