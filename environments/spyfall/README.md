# Spyfall Game Environment

Spyfall is a sophisticated social deduction game that evaluates language models' abilities in strategic deception, communication, and reasoning. It provides a rich testing ground for measuring both tactical intelligence and social interaction capabilities of LLMs within a competitive yet collaborative scenario.

## Game Overview

In the PolitAgent benchmark framework, Spyfall serves as a controlled environment to measure how well language models can:

- Engage in strategic deception (as the spy)
- Detect deceptive behavior (as villagers)
- Balance information disclosure
- Interpret ambiguous statements
- Make decisions based on incomplete information
- Reason about other agents' knowledge and intentions

### Core Game Mechanics

- **Setup**: 6 language model agents participate, with one randomly assigned as the spy
- **Information Asymmetry**: Villagers receive a common word (e.g., "planet"), while the spy receives a related but different word (e.g., "moon")
- **Deception Challenge**: The spy must pretend to know the villagers' word without giving away their ignorance
- **Detection Challenge**: Villagers must identify inconsistencies in descriptions to find the spy
- **Dynamic Rounds**: Play proceeds through multiple rounds of description and voting until a termination condition
- **Social Calibration**: Optimal strategy requires balancing specificity - being too vague or too specific can expose a player

## Strategic Dimensions Measured

Spyfall specifically tests several key capabilities in language models:

### 1. Strategic Deception Skills

For the spy agent, success requires:
- **Adaptive Vagueness**: Crafting descriptions that are neither too vague nor too specific
- **Contextual Learning**: Using information from other players to inform their own deception
- **Confidence Calibration**: Projecting appropriate confidence despite uncertainty
- **Information Extraction**: Gathering clues about the actual word from others' descriptions

### 2. Detection and Analytical Reasoning

For villager agents, success requires:
- **Inconsistency Detection**: Identifying subtle discrepancies in others' descriptions
- **Bayesian Updating**: Revising beliefs about player roles as new evidence emerges
- **Pattern Recognition**: Distinguishing genuine from artificial familiarity
- **Strategic Voting**: Coordinating votes to maximize spy detection probability

### 3. Communication and Social Dynamics

For all agents, the game tests:
- **Pragmatic Communication**: Conveying appropriate levels of information
- **Theory of Mind**: Reasoning about others' knowledge states
- **Strategic Disclosure**: Deciding what information to reveal and conceal
- **Meta-reasoning**: Thinking about how one's statements will be interpreted

## Comprehensive Metrics System

Spyfall employs a sophisticated metrics framework to evaluate agent performance across multiple dimensions:

### Core Performance Metrics

- **Spy Detection Rate**: Percentage of games where villagers correctly identify the spy
- **Spy Survival Rate**: Percentage of games where the spy avoids detection
- **Rounds to Resolution**: Average number of rounds before game conclusion
- **Vote Accuracy**: How often players vote for the actual spy
- **Efficient Elimination**: Whether non-spy players are correctly preserved

### Advanced Analytical Metrics

- **Description Specificity**: Quantitative measurement of how specific/vague descriptions are
- **Perplexity Analysis**: Evaluates linguistic consistency of descriptions using language model perplexity
- **Vagueness Scoring**: Algorithmic assessment of ambiguity in player statements
- **Chain-of-Thought Coherence**: Measures logical consistency in agent reasoning processes
- **Vote Influence Index**: Tracks how players' votes affect others' subsequent decisions

### Strategic Evaluation Metrics

- **Deception Effectiveness**: How well spies blend in with villagers' descriptions
- **Information Extraction**: How efficiently players gather useful information
- **Strategic Adaptation**: How agents adjust strategies based on game progression
- **Rational Play Score**: Deviation from theoretically optimal play given information available
- **Tactical Decision Quality**: Evaluation of key decision points against optimal choices

### Meta-Game Analysis

- **Cross-Model Performance**: Comparative analysis of different LLMs in same game scenarios
- **Role-Based Advantage**: Whether certain models perform better as spies or villagers
- **Adversarial Robustness**: Performance against strategic opponents
- **Meta-Strategy Evolution**: How strategies evolve over repeated games

## LLM Capabilities Assessment

Through the Spyfall benchmark, we can assess several critical capabilities of language models:

### 1. Pragmatic Communication

The game tests whether models can:
- Generate contextually appropriate descriptions
- Balance specificity and vagueness strategically
- Adjust communication based on their role and game state
- Avoid revealing too much or too little information

### 2. Strategic Reasoning

Models must demonstrate:
- Forward-thinking about how their statements will affect future game states
- Counterfactual reasoning about what would happen under different scenarios
- Risk assessment when deciding how specific to make descriptions
- Multi-level strategic thinking (reasoning about others reasoning about them)

### 3. Social Intelligence

The benchmark evaluates:
- Theory of Mind capabilities (understanding others' knowledge states)
- Intention reading from subtle linguistic cues
- Social calibration (not appearing too suspicious or too obvious)
- Group dynamic navigation and reputation management

### 4. Adversarial Robustness

The game provides insights into:
- How models perform against deceptive information
- Detection capabilities for inconsistencies and falsehoods
- Resilience against manipulation attempts
- Balance between trust and skepticism

## Political Agent Implications

The Spyfall environment provides valuable insights into LLM capabilities relevant for political agent simulation:

### Strategic Communication

- **Information Control**: How agents strategically share or withhold information
- **Selective Disclosure**: Revealing enough to build trust while concealing strategically important details
- **Coalition Formation**: How shared information creates implicit alliances among players
- **Persuasive Signaling**: Using specific language to signal group membership or knowledge

### Trust Dynamics

- **Trust Building**: How consistent, helpful information establishes credibility
- **Trust Verification**: How agents verify claims through cross-referencing
- **Reputation Effects**: How past behavior influences current trust levels
- **Deception Detection**: How subtle inconsistencies trigger suspicion

### Group Decision Making

- **Consensus Building**: How groups arrive at decisions about who to vote for
- **Information Cascades**: How early votes influence later ones
- **Misinformation Effects**: How incorrect information propagates through the group
- **Strategic Voting**: Balancing individual assessment with group influence

### Institutional Design Implications

- **Information Environment**: How the structure of available information affects decision quality
- **Voting Mechanisms**: How different voting procedures affect outcomes
- **Deliberation Effects**: How discussion rounds improve or degrade decision accuracy
- **Group Size Effects**: How performance scales with different numbers of players

## Technical Implementation

### File Structure

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

### Metrics Implementation

The metrics system integrates with the base PolitAgent framework and covers:

1. **Event-Based Recording**: Game events are recorded through a standardized system
2. **Multi-level Analysis**: Metrics are calculated at the turn, round, and game levels
3. **Model Performance Tracking**: Usage statistics and response quality are measured
4. **Strategic Evaluation**: Specialized metrics for spy deception and villager detection
5. **LLM-as-Judge**: Optional evaluation using a separate LLM to assess gameplay quality

### Game Flow

1. **Initialization**:
   - Set up the environment with specified models
   - Load word pairs and assign roles
   - Initialize recording metrics

2. **Game Loop**:
   - Describing Stage: Each player describes their word
   - Voting Stage: Players vote on suspected spy
   - Elimination: Voted player is removed from the game
   - Termination Check: End conditions are evaluated

3. **Resolution**:
   - Winner is determined (spy or villagers)
   - Comprehensive metrics are calculated
   - Results are stored for analysis

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

### Full Benchmark Mode

```bash
python -m core.benchmark --full_benchmark --models openai --specific_model gpt-4
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
| `--use_llm_evaluation` | Enable LLM-as-judge | `False` |
| `--evaluation_model` | Model for game evaluation | `None` (uses spy model) |

## Research Applications

The Spyfall environment serves as an experimental platform for research into:

1. **Cooperative AI**: Testing multi-agent coordination in mixed-motive settings
2. **LLM Alignment**: Evaluating models' adherence to human-like social norms
3. **Deception Detection**: Developing methods to identify AI-generated deceptive content
4. **Group Decision-Making**: Understanding how information aggregation affects outcomes
5. **Strategic Communication**: Analyzing information exchange under competitive pressures
6. **Theory of Mind**: Measuring LLMs' ability to model others' mental states
7. **Misinformation Dynamics**: Studying how false information propagates in groups

## Analysis Examples

The environment produces rich datasets for analysis, including:

- Visualization of player suspicion networks as they evolve over time
- Linguistic analysis of deceptive vs. truthful descriptions
- Decision trees showing critical voting patterns
- Comparison of different LLM capabilities in strategic environments
- Ablation studies on different prompt designs and their effects

By systematically varying game parameters and model types, researchers can isolate specific capabilities and limitations of language models in strategic social settings.