# PolitAgent Benchmark - Advanced LLM Strategic Reasoning and Social Dynamics Evaluation Framework

PolitAgent Benchmark is a comprehensive evaluation framework for assessing Large Language Models (LLMs) in complex multi-agent scenarios. It provides standardized environments for measuring strategic reasoning, deception detection, social coordination, information manipulation, and political decision-making capabilities.

## Table of Contents

- [Overview](#overview)
- [Evaluation Framework](#evaluation-framework)
- [Metrics System](#metrics-system)
- [LLM Capabilities Assessment](#llm-capabilities-assessment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Supported Models](#supported-models)
- [Supported Games](#supported-games)
- [Running Benchmarks](#running-benchmarks)
- [Analyzing Results](#analyzing-results)
  - [Single Benchmark Analysis](#single-benchmark-analysis)
  - [Comprehensive Multi-Benchmark Analysis](#comprehensive-multi-benchmark-analysis)
  - [Scoring Methodology](#scoring-methodology)
- [Extending the Framework](#extending-the-framework)
  - [Adding a New Game](#adding-a-new-game)
  - [Adding a New Model](#adding-a-new-model)
- [Using Ollama Models](#using-ollama-models)
- [License](#license)

## Overview

The PolitAgent Benchmark creates controlled environments where language models must engage in strategic interactions, reason about hidden information, navigate deception, and coordinate with or compete against other agents. By placing LLMs in these challenging social dynamics, the framework reveals their capabilities and limitations in situations requiring advanced reasoning and strategic thinking.

Key features include:
- Multiple game environments with diverse strategic challenges
- Standardized metrics for cross-model comparison
- Comprehensive performance analytics
- Multi-agent interactions with varying incentive structures
- Support for all major LLM providers and local models
- Extensible architecture for adding new games and metrics

## Evaluation Framework

The PolitAgent Benchmark evaluates language models across five critical domains:

### 1. Strategic Reasoning

- **Long-term Planning**: Ability to develop multi-step strategies toward objectives
- **Decision Making Under Uncertainty**: Reasoning effectively with incomplete information
- **Bayesian Updating**: Appropriately revising beliefs based on new evidence
- **Game-Theoretic Understanding**: Recognition of strategic equilibria and optimal plays
- **Counterfactual Analysis**: Consideration of alternative action paths
- **Strategic Adaptation**: Adjusting plans based on changing circumstances
- **Resource Optimization**: Efficient allocation of limited resources (questions, actions, etc.)

### 2. Social Intelligence

- **Theory of Mind**: Understanding and modeling other agents' knowledge and beliefs
- **Intention Recognition**: Inferring others' plans from their actions and communications
- **Cooperative Coordination**: Aligning actions with allies toward common goals
- **Trust Calibration**: Appropriately adjusting trust levels based on observed behavior
- **Coalition Formation**: Building and maintaining strategic partnerships
- **Social Positioning**: Managing relationships within complex social networks
- **Reputation Management**: Building and leveraging credibility strategically

### 3. Information Operations

- **Strategic Information Gathering**: Asking optimal questions to reduce uncertainty
- **Deception Detection**: Identifying false or misleading statements
- **Misinformation Analysis**: Evaluating the credibility of contradictory information
- **Strategic Disclosure**: Controlling what information to reveal or withhold
- **Persuasive Communication**: Crafting arguments that influence others' decisions
- **Lie Crafting**: Creating plausible deceptions when strategically advantageous
- **Information Integration**: Synthesizing partial information into coherent theories

### 4. Political Reasoning

- **Power Dynamics Analysis**: Understanding influence structures in multi-agent systems
- **Coalition Politics**: Managing alliance structures across competing interests
- **Narrative Control**: Shaping collective understanding through strategic communication
- **Credible Commitment Problems**: Navigating trust issues in non-binding agreements
- **Collective Action Coordination**: Organizing group efforts toward common objectives
- **Strategic Communication**: Using language to achieve specific political goals
- **Policy Formulation**: Developing rules that align incentives toward desired outcomes

### 5. Cognitive Performance

- **Logical Consistency**: Maintaining coherent reasoning across multiple interactions
- **Constraint Satisfaction**: Finding solutions that satisfy multiple competing conditions
- **Memory Utilization**: Effectively tracking and using information from previous interactions
- **Adaptive Decision-Making**: Appropriately responding to changes in strategic environment
- **Strategic Complexity Management**: Handling multi-level, nested strategic problems
- **Cognitive Bias Management**: Avoiding common reasoning errors under strategic pressure
- **Metacognitive Awareness**: Understanding limits of one's knowledge and reasoning

## Metrics System

The PolitAgent Benchmark employs a sophisticated multi-layered metrics system:

### Core Infrastructure

- **Event-Based Recording**: All game events are captured in a standardized format
- **Cross-Game Compatibility**: Metrics can be compared across different environments
- **Multi-Level Analysis**: Metrics are calculated at turn, round, and game levels
- **Performance Aggregation**: Results can be aggregated across multiple runs
- **Model Comparison**: Standardized metrics enable direct comparison between different LLMs
- **Statistical Significance Testing**: Built-in tools for determining meaningful differences

### Universal Metrics

Applied across all game environments:

- **Decision Quality**: Assessment of strategic optimality of decisions
- **Consistency Score**: Measurement of logical coherence across multiple decisions
- **Success Rate**: Percentage of games where objectives are achieved
- **Reasoning Chain Quality**: Evaluation of thinking process quality
- **Context Utilization**: How effectively available information is used
- **Adaptive Response**: How well models adjust to unexpected situations
- **Communication Effectiveness**: Quality and strategic value of generated messages

### Game-Specific Metrics

Each game environment implements specialized metrics:

#### Spyfall
- Detection accuracy, strategic deception, information management, social calibration, etc.

#### Beast
- Strategic negotiation, coalition formation, betrayal timing, resource management, etc.

#### AskGuess
- Information gain per question, hypothesis testing, uncertainty reduction, etc.

#### TofuKingdom
- Deception detection, logical consistency, team coordination, role identification, etc.

#### Diplomacy
- Alliance stability, negotiation success, territorial control, strategic positioning, etc.

### LLM-as-Judge Evaluation

The framework includes an optional "LLM-as-judge" component that provides:
- Expert assessment of strategic play quality
- Comparative analysis of decision points
- Identification of critical errors and missed opportunities
- Recommendations for strategic improvement
- Detailed performance analysis reports

## LLM Capabilities Assessment

Through comprehensive benchmarking, PolitAgent reveals critical aspects of LLM capabilities:

### Strategic Depth

- **Horizon Length**: How many steps ahead models can effectively plan
- **Search Space Navigation**: How efficiently models explore possible strategies
- **Strategic Originality**: Development of novel approaches to problems
- **Robustness to Deception**: Performance when faced with intentional misinformation
- **Risk Assessment**: Accurate evaluation of strategic risks and rewards
- **Decision Quality Under Pressure**: Performance with limited time/information

### Social Understanding

- **Agent Modeling**: Accuracy in predicting other agents' behavior
- **Social Calibration**: Appropriateness of social interactions in different contexts
- **Coalition Management**: Effectiveness in building and maintaining alliances
- **Motivational Understanding**: Recognition of others' incentives and goals
- **Adaptation to Betrayal**: Recovery after trust violations
- **Multi-agent Coordination**: Success in collective action scenarios

### Political Intelligence

- **Power Recognition**: Understanding of influence dynamics in multi-agent systems
- **Narrative Construction**: Creation of persuasive explanatory frameworks
- **Strategic Positioning**: Optimal placement within social networks
- **Information Warfare**: Effectiveness in controlling and countering narratives
- **Coalition Leadership**: Success in directing aligned groups
- **Opposition Management**: Strategies for dealing with competing interests

### Cognitive Architecture Insights

- **Memory Integration**: How models incorporate past interactions into current decisions
- **Belief Updating**: How models revise their understanding based on new information
- **Strategic Focus**: Ability to identify critical decision points
- **Contradiction Handling**: Resolution of inconsistent or paradoxical information
- **Uncertainty Management**: Effective decision-making despite incomplete information
- **Counterfactual Reasoning**: Quality of "what if" scenario analysis

## Project Structure

```
PolitAgent-environments/
├── core/
│ ├── benchmark.py # Main benchmarking script
│ ├── benchmark_visualizer.py # Results analysis and visualization
│ └── init.py
├── environments/
│ ├── spyfall/ # Social deduction with hidden roles and information asymmetry
│ ├── beast/ # Strategic negotiation and coalition formation
│ ├── askguess/ # Optimal information gathering and hypothesis testing
│ ├── tofukingdom/ # Logic puzzles with deception and team dynamics
│ └── diplomacy_game/ # Complex negotiation and territorial strategy
├── metrics/
│ ├── base_metrics.py # Common metrics infrastructure
│ ├── spyfall_metrics.py # Spyfall-specific metrics
│ ├── beast_metrics.py # Beast-specific metrics
│ ├── askguess_metrics.py # AskGuess-specific metrics
│ ├── tofukingdom_metrics.py # TofuKingdom-specific metrics
│ └── diplomacy_metrics.py # Diplomacy-specific metrics
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

### 5. T2L (Text-to-LoRA) Setup

To enable T2L functionality in `llm/t2l_chat.py`, follow these additional steps:

#### Install T2L Dependencies

After updating `pyproject.toml` with T2L dependencies, update your environment:

```bash
poetry lock
poetry install
```

#### Download Pre-trained T2L Models

T2L requires pre-trained checkpoints. Download them using the Hugging Face CLI:

```bash
# Install huggingface-hub CLI (already included in dependencies)
huggingface-cli download microsoft/HyperLoRA-Llama3.1-8B-Instruct --local-dir ./trained_t2l/llama3.1-8b-instruct
huggingface-cli download microsoft/HyperLoRA-Gemma2-9B-Instruct --local-dir ./trained_t2l/gemma2-9b-instruct
```

#### Hardware Requirements

- **GPU Memory**: At least 16GB VRAM recommended for T2L inference
- **System RAM**: 32GB+ recommended for large model loading
- **Storage**: ~20GB for downloaded T2L checkpoints

#### T2L Integration

The `T2LChatModel` class in `llm/t2l_chat.py` provides:

- **Dynamic LoRA Generation**: Creates task-specific adapters on-the-fly
- **Multi-Tool Support**: Includes `T2LActTool`, `T2LTalkTool`, and `T2LSelfQuestionTool`
- **LangChain Integration**: Compatible with existing agent frameworks
- **Diplomatic Actions**: Specialized tools for negotiation and strategic communication

#### Usage Example

```python
from llm.t2l_chat import T2LChatModel

# Initialize T2L model with checkpoint path
t2l_model = T2LChatModel(
    checkpoint_path="./trained_t2l/llama3.1-8b-instruct",
    model_name="llama3.1-8b-instruct"
)

# Use in agent-based scenarios
agent = t2l_model.create_agent()
response = agent.invoke({"input": "Negotiate a trade agreement"})
```

#### Troubleshooting

- **CUDA Issues**: Ensure PyTorch is installed with CUDA support
- **Memory Errors**: Reduce batch size or use model quantization
- **Download Failures**: Check internet connection and Hugging Face access

For more details, see the T2L documentation in the `text-to-lora/` directory.

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
A social deduction game measuring strategic deception, information management, and collective intelligence. One agent must pretend to know the secret word while avoiding detection, while others must identify the impostor through careful questioning and observation.

**Key Capabilities Tested:**
- Strategic communication under information asymmetry
- Deception generation and detection
- Social calibration and reputation management
- Group decision-making dynamics

### 2. Beast
A strategic negotiation game assessing coalition formation, resource management, and political maneuvering. Players engage in strategic alliances, betrayals, and resource competition through multiple phases of complex interaction.

**Key Capabilities Tested:**
- Multi-agent negotiation and bargaining
- Trust building and strategic betrayal
- Resource allocation and risk assessment
- Alliance formation and maintenance

### 3. AskGuess
An information elicitation game evaluating question optimization, hypothesis formation, and deductive reasoning. One agent must discover a hidden concept by asking strategic questions to maximize information gain.

**Key Capabilities Tested:**
- Strategic question formulation
- Information theory understanding
- Hypothesis generation and testing
- Efficient search space reduction

### 4. TofuKingdom
A logical deduction game measuring truth discrimination, role identification, and reasoning under deception. Players have conflicting incentives to either reveal or conceal information based on team affiliations.

**Key Capabilities Tested:**
- Logical reasoning with contradictory information
- Deception detection in structured environments
- Team-based strategic coordination
- Decision-making with partial information

### 5. Diplomacy
A geopolitical strategy game assessing negotiation, alliance management, and territorial control. Players engage in complex diplomatic negotiations while managing military units across a European map.

**Key Capabilities Tested:**
- Strategic negotiation and alliance formation
- Commitment problems in non-binding agreements
- Complex spatial and temporal planning
- Balancing cooperation and competition

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

### Full Benchmark Suite

```bash
python -m core.benchmark --full_benchmark --models openai --specific_model gpt-4
```

#### Common Arguments

- `--models`: Comma-separated list of models (e.g., `openai,mistral,vllm`)
- `--games`: Comma-separated list of games (`spyfall,beast,askguess,tofukingdom,diplomacy_game`)
- `--workers`: Number of parallel processes
- `--runs_per_game`: Number of runs per game/phrase
- `--debug`: Enable verbose logging
- `--max_phrases`: Limit number of phrases/labels to process (useful for testing)
- `--use_llm_evaluation`: Enable LLM-as-judge for advanced metric evaluation
- `--evaluation_model`: Specify the model to use for LLM-as-judge evaluation

#### Game-Specific Arguments

**Spyfall:**
- `--label_path`: Path to word pairs file
- `--spy_model_name`: Model for the spy
- `--villager_model_name`: Model for villagers
- `--openai_api_key`: OpenAI API key (overrides env variable)
- `--embedding_model`: Embedding model type (`local`, `openai`, `auto`)
- `--embedding_model_name`: Embedding model name
- `--perplexity_model`: Perplexity model (`auto`, `local`, or model name)

**Beast:**
- `--model_name`: Model name
- `--num_players`: Number of players (6-8)
- `--max_rounds`: Number of rounds (5-8)

**AskGuess:**
- `--model_name`: Model name
- `--mode`: Game mode (`easy` or `hard`)
- `--max_rounds`: Maximum questions allowed

**TofuKingdom:**
- `--prince_model_name`: Model for prince
- `--princess_model_name`: Model for princess
- `--queen_model_name`: Model for queen
- `--neutral_model_name`: Model for neutral character

**Diplomacy:**
- `--max_rounds`: Maximum game rounds
- `--model_name`: Model to use for all powers

## Analyzing Results

### Single Benchmark Analysis

Analyze results from a single benchmark run:

```bash
python -m core.benchmark_analyzer --results_dir benchmark_results/20240529_123456
```

### Comprehensive Multi-Benchmark Analysis

Analyze results across multiple benchmark runs for comprehensive model evaluation:

```bash
python -m core.comprehensive_benchmark_analyzer --benchmark_dirs benchmark_results/20240529_123456 benchmark_results/20240530_234567 benchmark_results/20240531_345678
```

### Scoring Methodology

The PolitAgent benchmark uses a sophisticated weighted scoring system to evaluate model performance:

#### Scoring Formula
```
Final Score = Σ[(success_rate × 0.7 + efficiency × 0.3) × complexity_weight] / Σ[complexity_weights]
```

#### Game Complexity Weights
- **AskGuess**: 1.0x (basic information gathering and deduction)
- **Spyfall**: 1.0x (social deduction and role consistency)
- **Beast**: 1.0x (strategic negotiation and survival)
- **TofuKingdom**: 1.2x (logic puzzles with deception detection)
- **Diplomacy**: 1.5x (complex multi-agent geopolitical strategy)

#### Component Metrics
- **Success Rate** (70% weight): Percentage of games where the model achieved its objective
- **Efficiency** (30% weight): How quickly/optimally the model achieved success (when available)

#### Performance Classification
- **🟢 Excellent** (0.8-1.0): Exceptional strategic thinking across environments
- **🟡 Good** (0.6-0.8): Solid performance with room for improvement  
- **🟠 Fair** (0.4-0.6): Basic competency, struggles with advanced reasoning
- **🔴 Poor** (0.2-0.4): Limited understanding of game mechanics
- **⚫ Very Poor** (0.0-0.2): Requires significant improvement

### Analysis Output

The analyzers generate comprehensive reports including:

1. **Executive Summary**: Overall score, performance classification, and total games analyzed
2. **Game-by-Game Breakdown**: Detailed metrics for each game environment
3. **Scoring Methodology**: Transparent explanation of how scores are calculated
4. **Performance Assessment**: Qualitative evaluation of model capabilities
5. **Improvement Recommendations**: Specific suggestions for enhancing model performance
6. **Detailed Game Results**: Individual game outcomes and key decision points

#### Output Formats
- **Markdown Reports**: Human-readable analysis with detailed breakdowns
- **JSON Data**: Machine-readable results for further analysis
- **Console Output**: Quick summary for immediate feedback

#### Example Analysis Commands

```bash
# Basic analysis with report output
python -m core.benchmark_analyzer --results_dir benchmark_results/20240529_123456 --output_file analysis_report.md

# JSON output for programmatic analysis
python -m core.benchmark_analyzer --results_dir benchmark_results/20240529_123456 --json_output results.json

# Comprehensive analysis across multiple runs
python -m core.comprehensive_benchmark_analyzer \
  --benchmark_dirs benchmark_results/run1 benchmark_results/run2 benchmark_results/run3 \
  --output_file comprehensive_report.md \
  --json_output comprehensive_results.json
```

## Extending the Framework

### Adding a New Game

1. Create a new directory under `environments/` with your game logic, agents, and prompts.
2. Create a metrics class in `metrics/` that extends `BaseMetrics`.
3. Implement a game class with `init_game` and `game_loop` methods.
4. Register your game in the `GAME_ENVIRONMENTS` dictionary in `core/benchmark.py`.
5. Add any custom metrics to `create_performance_dataframe` in `core/benchmark_visualizer.py`.

Key files to implement:
```
environments/your_game/
├── game.py              # Main game implementation
├── agents/              # Agent implementations
├── utils/               # Game-specific utilities
├── prompts/             # Prompt templates
└── README.md            # Documentation

metrics/
└── your_game_metrics.py # Game-specific metrics
```

### Adding a New Model

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

### Note on Performance

Local models through Ollama may have different performance characteristics compared to cloud models. Consider these tips:
- Start with smaller games like Beast or AskGuess
- Local models typically work better with fewer agents
- Adjust temperature settings if responses seem too random or too deterministic

## Research Applications

The PolitAgent Benchmark enables research into several critical areas:

### LLM Capabilities Assessment

- Systematic evaluation of strategic reasoning depth
- Identification of social intelligence limitations
- Measurement of theory of mind capabilities
- Testing of complex logical reasoning abilities
- Assessment of multi-step planning horizon

### Political Agent Development

- Training agents for complex negotiations
- Developing systems for coalition management
- Creating frameworks for information reliability assessment
- Building agents capable of deception detection
- Enhancing strategic decision-making capabilities

### Multi-Agent System Research

- Studying emergent behaviors in agent collectives
- Analyzing trust dynamics in strategic environments
- Exploring coalition formation and stability
- Investigating information cascade phenomena
- Testing coordination mechanisms under competing incentives

### Social Intelligence Research

- Measuring social reasoning capabilities
- Evaluating strategic communication effectiveness
- Assessing reputation management strategies
- Testing strategies for building trust
- Studying deception generation and detection

## Summary

The PolitAgent Benchmark provides a unified, comprehensive evaluation framework for assessing LLM capabilities in strategic environments. Key features include:

### Unified Scoring System
- **Consistent methodology** across all analyzers and game types
- **Transparent scoring formula** with detailed breakdowns
- **Weighted complexity scoring** recognizing different strategic challenges
- **Performance classification** with actionable recommendations

### Analysis Tools
- **Single Benchmark Analyzer** (`core.benchmark_analyzer`): Analyze individual benchmark runs
- **Comprehensive Analyzer** (`core.comprehensive_benchmark_analyzer`): Multi-run analysis for thorough model evaluation
- **Multiple output formats**: Markdown reports, JSON data, console summaries

### Model Evaluation Capabilities
- **Strategic reasoning assessment** across 5 game environments
- **Success rate and efficiency metrics** for performance measurement  
- **Detailed game-by-game analysis** with specific failure modes
- **Cross-game comparison** with standardized scoring
- **Improvement recommendations** based on performance patterns

The framework enables researchers and developers to systematically evaluate and improve LLM performance in complex multi-agent strategic scenarios.

## License

MIT

## Contact

For questions or contributions, please open an issue or pull request on GitHub.