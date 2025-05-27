# Beast Game Environment - Strategic Simulation Platform

Beast is a sophisticated social strategy game designed to evaluate language model capabilities in handling complex multi-agent interactions, strategic reasoning, information asymmetry, and political decision-making. This advanced simulation platform challenges LLMs to navigate ambiguous social dynamics, manage hidden information, and make consequential decisions under escalating pressure.

## Strategic Testing Framework

Beast serves as a controlled environment to rigorously assess language models' abilities in:

- **Strategic Deception & Detection**: Balancing information disclosure and uncovering others' hidden intentions
- **Coalition Formation & Betrayal**: Building temporary alliances while preparing for potential defection
- **Information Warfare**: Distributing selective information and countering misinformation
- **Multi-agent Coordination**: Negotiating complex multi-party agreements under conflicting incentives
- **Risk Assessment & Management**: Making decisions under uncertainty with incomplete information
- **Reputation Dynamics**: Managing perceived trustworthiness across sequential interactions
- **Resource Optimization**: Allocating limited resources efficiently across competing objectives

## Game Architecture

### Enhanced Game Mechanics

Beast implements a sophisticated multi-phase game structure:

#### 1. Intelligence Gathering Phase
- Players investigate others to discover hidden information
- Strategic spread of information and misinformation
- Resource allocation between information collection and security
- Trust level tracking across player interactions

#### 2. Alliance Formation Phase
- Secret coalition building with formalized agreements
- Strategic positioning within the social network
- Balance between commitment and flexibility
- Hidden alliance objectives and incentive structures

#### 3. Strategic Challenge Phase
- Asymmetric resource competitions
- Prisoner's dilemma and collective action problems
- Individual vs. group interest dynamics
- Variable-sum game scenarios with multiple equilibria

#### 4. Negotiation Phase
- Multi-party bargaining under time pressure
- Credible commitment mechanisms
- Information trading and strategic disclosure
- Complex offer evaluation and counterproposal dynamics

#### 5. Elimination Voting Phase
- Strategic voting with incomplete information
- Coalition coordination for voting blocs
- Reputation-based decision making
- Anticipation of future round implications

### Advanced Features

- **Hidden Information Architecture**: Layered information access with varying visibility
- **Trust System**: Dynamic trust metrics that evolve based on behavior
- **Multiple Resources**: Management of wealth, influence points, and information
- **Role-Based Asymmetry**: Special abilities that create strategic depth
- **Time Pressure**: Forcing quick decisions that reveal reasoning quality
- **Multi-Round Dynamics**: Actions in early rounds affect later strategic options

## Comprehensive Metrics System

Beast employs a sophisticated metrics framework to evaluate agent performance across multiple dimensions:

### 1. Model Performance Metrics

- **Inference Quality**: Evaluates raw response quality across different game phases
- **Decision Consistency**: Measures strategic coherence across multiple rounds
- **Context Utilization**: Assesses how effectively agents use available information
- **Error Analysis**: Tracks logical inconsistencies and decision flaws
- **Response Patterns**: Identifies characteristic reasoning approaches

### 2. Strategic Intelligence Assessment

- **Strategic Depth Index**: Measures multi-step planning capabilities
- **Adaptability Score**: Evaluates response to changing game conditions
- **Hidden Information Usage**: Assesses ability to operate with incomplete information
- **Counterfactual Reasoning**: Measures anticipation of others' potential actions
- **Equilibrium Analysis**: Evaluates convergence to game-theoretic optimal strategies

### 3. Social Intelligence Metrics

- **Alliance Stability**: Measures coalition formation effectiveness
- **Trust Cultivation**: Tracks ability to build and maintain trust
- **Betrayal Timing**: Analyzes optimal defection decision-making
- **Reputation Management**: Evaluates strategic identity projection
- **Social Network Position**: Measures centrality in alliance structures
- **Persuasion Effectiveness**: Quantifies success in changing others' decisions

### 4. Economic Performance Analysis

- **Wealth Accumulation**: Final and average wealth metrics
- **Transaction Efficiency**: Value creation through strategic exchanges
- **Resource Allocation**: Optimal distribution across competing needs
- **Risk-Adjusted Returns**: Performance accounting for strategic exposure

### 5. Political Agent Metrics

- **Influence Maximization**: Ability to shape group decisions
- **Coalition Leadership**: Effectiveness in directing alliance actions
- **Information Control**: Success in managing knowledge distribution
- **Public vs. Private Messaging**: Strategic differences in communication channels
- **Power Projection**: Capability to affect outcomes through reputation
- **Narrative Control**: Ability to frame explanations favorable to one's position

### 6. Behavioral Analysis

- **Risk Preference Profiling**: Reveals agent risk tolerance patterns
- **Reciprocity Patterns**: Measures response to cooperative/competitive actions
- **Truth-Telling Tendency**: Tracks honesty across different contexts
- **First-Mover Behavior**: Analyzes initiative-taking vs. reactive strategies
- **Learning Curve**: Measures adaptation to game dynamics over time

## LLM Capability Assessment

Through the Beast simulation platform, we can assess several critical capabilities of language models:

### 1. Strategic Reasoning

- **Multi-step Planning**: Ability to reason across sequential rounds
- **Bayesian Updating**: Appropriately revising beliefs with new evidence
- **Game-Theoretic Understanding**: Recognizing and exploiting strategic equilibria
- **Hidden Information Reasoning**: Operating effectively with partial visibility
- **Counterfactual Analysis**: Exploring alternative decision paths

### 2. Social Intelligence

- **Theory of Mind**: Understanding others' knowledge states and intentions
- **Alliance Psychology**: Managing group dynamics and coalition formation
- **Trust Calibration**: Appropriate trust levels for different agents
- **Reputation Dynamics**: Understanding how actions affect perceived trustworthiness
- **Strategic Empathy**: Using understanding of others' motivations for advantage

### 3. Political Reasoning

- **Power Dynamics**: Understanding formal and informal influence structures
- **Coalition Politics**: Managing alliances and voting blocs
- **Strategic Communication**: Selective information sharing for advantage
- **Legitimacy Building**: Creating narratives that justify actions
- **Crisis Management**: Optimal responses under elimination pressure

### 4. Decision Quality Under Uncertainty

- **Risk Assessment**: Accurately evaluating probabilistic outcomes
- **Time-Pressure Performance**: Maintaining reasoning quality under constraints
- **Information Valuation**: Correctly prioritizing different types of knowledge
- **Decision Consistency**: Avoiding contradictory or incoherent strategies
- **Adaptive Decision-Making**: Adjusting strategies as game state evolves

## Research Applications

The Beast environment serves as an experimental platform for research into:

### Political Agent Development

- **Strategic Interaction**: Testing multi-agent coordination in mixed-motive settings
- **Information Asymmetry**: Evaluating performance under variable information conditions
- **Coalition Dynamics**: Studying alliance formation and maintenance under stress
- **Reputation Systems**: Developing robust models of trust and betrayal
- **Information Warfare Resilience**: Testing resistance to strategic misinformation

### LLM Capability Assessment

- **Strategic Depth**: Measuring multi-round planning capabilities
- **Social Intelligence**: Evaluating understanding of complex social dynamics
- **Economic Reasoning**: Testing resource optimization under constraints
- **Theory of Mind**: Assessing models' ability to represent others' mental states
- **Decision Quality**: Evaluating choices under uncertainty and time pressure

### Political Theory Applications

- **Institutional Design**: Testing how rule structures affect outcomes
- **Voting System Analysis**: Comparing different voting mechanisms
- **Information Environment Effects**: Studying how information availability shapes decisions
- **Power Distribution**: Analyzing formal and informal influence structures
- **Coalition Stability**: Investigating factors that strengthen or weaken alliances

## Technical Implementation

### File Structure

```
beast/
├── game.py               # Enhanced strategic game implementation
├── agents/               # Agent implementations with strategic capabilities
│   ├── base_agent.py     # Advanced agent with multi-phase decision logic
│   └── __init__.py       # Package exports
├── utils/                # Utility functions
│   ├── prompt.py         # Strategic prompt engineering templates
│   └── utils.py          # Helper functions for game mechanics
├── prompts/              # Specialized prompt templates for each game phase
└── __init__.py           # Package exports
```

### Metrics Implementation

The metrics system integrates with the PolitAgent framework and provides:

1. **Multi-dimensional Analysis**: Performance metrics across strategic, social, economic dimensions
2. **Phase-Specific Evaluation**: Targeted metrics for each game phase
3. **Comparative Analysis**: Cross-model performance comparison
4. **Behavioral Profiling**: Identification of strategic tendencies and patterns
5. **LLM-as-Judge**: Optional evaluation using an external model to assess gameplay quality

### Game Flow

1. **Initialization**:
   - Create players with randomized wealth, roles, and special abilities
   - Initialize trust matrix and information tracking systems

2. **Game Loop (each round)**:
   - Intelligence Gathering: Information collection and misinformation spread
   - Alliance Formation: Secret coalition building and strategy coordination
   - Strategic Challenge: Asymmetric resource competitions
   - Negotiation: Multi-party bargaining under time pressure
   - Elimination Voting: Strategic voting with coalition coordination
   - Update game state and apply pressure escalation

3. **Game Resolution**:
   - Final wealth determination
   - Victory condition evaluation
   - Comprehensive metrics calculation

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

### Full Benchmark Mode

```bash
python -m core.benchmark --full_benchmark --games beast --models openai --specific_model gpt-4
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | LLM provider to use | `openai` |
| `--output_dir` | Directory for saving game results | `./results/beast` |
| `--max_rounds` | Number of elimination rounds | `5` |
| `--num_players` | Number of players in the game | `6` |
| `--debug` | Enable verbose logging | `False` |
| `--use_llm_evaluation` | Enable LLM-as-judge evaluation | `False` |

## Analysis Examples

The environment produces rich datasets that enable:

1. **Strategic Profile Analysis**: Identifying characteristic reasoning patterns
2. **Social Network Visualization**: Mapping alliance structures and influence flows
3. **Economic Performance Comparisons**: Quantifying resource optimization capabilities
4. **Decision Quality Assessment**: Evaluating optimality of key strategic choices
5. **Multi-Agent Dynamics**: Studying emergent patterns in complex interactions

By systematically varying game parameters and model types, researchers can isolate specific capabilities and limitations of language models in strategic political environments.

## Political Agent Implications

The Beast environment offers valuable insights for political agent development:

### 1. Strategic Communication

- **Information Control**: How agents strategically share or withhold information
- **Selective Disclosure**: Revealing enough to build trust while concealing key details
- **Coalition Signaling**: Using specific language to coordinate with allies
- **Public vs. Private Communication**: Strategic differences in messaging channels

### 2. Trust and Power Dynamics

- **Trust Building**: How consistent, helpful actions establish credibility
- **Power Projection**: Using reputation to influence outcomes without direct action
- **Legitimacy Construction**: Creating narratives that justify strategic positions
- **Authority Challenges**: How coalitions form to counter dominant players

### 3. Institutional Design

- **Voting System Effects**: How different voting mechanisms affect outcomes
- **Information Access Rules**: How visibility constraints shape strategic options
- **Resource Distribution**: How initial allocations influence power dynamics
- **Time Pressure Effects**: How decision quality changes under constraints

### 4. Political Psychology

- **Risk Attitudes**: How uncertainty affects strategic decisions
- **In-Group/Out-Group Dynamics**: How alliance structures affect perception
- **Reciprocity Norms**: Patterns of cooperation and retaliation
- **Attribution Biases**: How agents explain others' actions based on limited information

Through systematic experimentation in the Beast environment, researchers can develop more sophisticated political agents capable of navigating the complex strategic terrain of multi-agent systems with incomplete information and mixed incentives. 