# TofuKingdom Game Environment - Strategic Deception Simulation

TofuKingdom is a sophisticated social deduction game that evaluates language models' abilities in strategic reasoning, deception detection, and political maneuvering within a medieval court intrigue scenario. This environment creates a complex network of competing incentives where models must navigate through intentional misinformation to achieve their objectives.

## Strategic Testing Framework

The TofuKingdom environment rigorously assesses language models' capabilities in:

- **Deception Detection**: Identifying false statements among conflicting testimonies
- **Strategic Questioning**: Formulating inquiries that extract maximum information value
- **Truth Discrimination**: Differentiating between truthful and deceptive statements
- **Role Identification**: Mapping behavioral patterns to underlying identities
- **Team Awareness**: Understanding coalition dynamics and competing incentives
- **Constrained Decision-Making**: Operating within fixed action parameters
- **Evidence Synthesis**: Compiling consistent theories from contradictory information

## Game Architecture

### Core Game Mechanics

TofuKingdom implements a structured social deduction framework:

#### 1. Role & Identity System
- 8 character roles with distinct truth-telling behaviors
- 3 competing teams with different victory conditions
- Hidden identities requiring strategic information gathering
- Asymmetric access to knowledge about true roles

#### 2. Truth Behavior Rules
- **Absolute Truth-Tellers**: Princess and Chef always tell the truth
- **Absolute Liars**: Queen, Minister, and Guard always lie
- **Strategic Actors**: Maid and Spy can choose to either lie or tell the truth
- **Information Seeker**: Prince must determine which player is the Princess

#### 3. Team Structure
- **Princess Team**: Princess and Chef (win if Prince correctly identifies Princess)
- **Queen Team**: Queen, Minister, and Guard (win if Prince mistakenly identifies Queen as Princess)
- **Neutral Team**: Maid and Spy (win if Prince chooses any team member besides Princess or Queen)

#### 4. Question Framework
- Restricted to three predefined question formats:
  1. "Which player is the Princess?"
  2. "What is your identity?"
  3. "What is the identity of {player_name}?"
- Sequential questioning of all players
- Final decisive question to one selected player
- Limited information-gathering opportunities

### Advanced Features

- **Logical Paradox Generation**: The lying mechanism creates classic liar paradoxes
- **Coalition Dynamics**: Team members act in coordinated self-interest
- **Adversarial Testimony**: Players provide information designed to mislead or misdirect
- **Stance Consistency Tracking**: Analysis of internal consistency in responses
- **Competing Theories**: Multiple possible interpretations of the evidence
- **Deception Layering**: Strategic lies built upon other strategic lies
- **Truth Mixing**: Partial truths combined with falsehoods for credibility

## Comprehensive Metrics System

TofuKingdom employs a sophisticated metrics framework to evaluate agent performance:

### 1. Strategic Questioning Metrics

- **Question Diversity**: Measures range of question types and targets
- **Information Value**: Evaluates how much each question reduces uncertainty
- **Targeting Strategy**: Assesses pattern of player selection for questioning
- **Follow-up Coherence**: Analyzes how questions build on previous information
- **Critical Question Identification**: Measures ability to ask decisive questions
- **Question Balance**: Evaluates distribution of question types
- **Question Sequencing**: Analyzes the logical progression of inquiry

### 2. Deception Detection Metrics

- **Lie Recognition Rate**: Percentage of lies correctly identified
- **Truth Recognition Rate**: Percentage of truths correctly identified
- **Confusion Matrix Analysis**: Patterns of truth/lie classification errors
- **Contradiction Identification**: Ability to detect logically incompatible statements
- **Logical Consistency Tracking**: Following the implications of truth/lie behaviors
- **Team Affiliation Detection**: Identifying team allegiances from behavioral patterns
- **Pattern Recognition**: Noticing consistent behavior across multiple responses

### 3. Role Performance Metrics

- **Truth Behavior Adherence**: How consistently roles follow their truth behavior rules
- **Strategic Effectiveness**: How well each role advances their team's goals
- **Answer Quality**: Sophistication and persuasiveness of responses
- **Deception Effectiveness**: For roles that lie, how convincing their lies are
- **Team Coordination**: How well team members align their strategies
- **Self-Preservation**: How effectively roles avoid detection
- **Tactical Misdirection**: Success in leading the Prince toward incorrect conclusions

### 4. Prince Performance Metrics

- **Correct Identification**: Whether the Prince correctly identifies the Princess
- **Question Strategy Effectiveness**: Quality of information gained through questions
- **Information Integration**: Ability to synthesize information across answers
- **Logical Reasoning**: Quality of deductive process from contradictory information
- **Truth Table Analysis**: Systematic testing of possible role configurations
- **Decision Confidence**: Certainty level in final Princess identification
- **Reasoning Chain Quality**: Logical path from evidence to conclusion

### 5. Game Outcome Metrics

- **Team Victory Distribution**: Percentage of games won by each team
- **Role Identification Accuracy**: Accuracy of Prince's beliefs about all roles
- **Critical Decision Points**: Key moments that determined game outcomes
- **Win Path Analysis**: Patterns in successful strategies for each team
- **Model Performance Comparison**: Relative success rates across different models
- **Game Balance Assessment**: Fairness of outcomes across team configurations
- **Decision Quality**: Relationship between reasoning process and correct outcomes

## LLM Capability Assessment

Through the TofuKingdom benchmark, we can assess several critical capabilities of language models:

### 1. Logical Reasoning

- **Syllogistic Reasoning**: Ability to follow chains of logical implications
- **Contradiction Resolution**: Handling inconsistencies in collected information
- **Constraint Satisfaction**: Finding solutions that satisfy observed truth behaviors
- **Impossible Case Elimination**: Ruling out logically impossible configurations
- **Propositional Logic**: Understanding how statements relate when inverted by lying
- **Truth Table Construction**: Systematically exploring possible configurations
- **Proof by Contradiction**: Testing hypotheses against established constraints

### 2. Strategic Intelligence

- **Information Source Evaluation**: Assessing credibility of different players
- **Theory Formation and Testing**: Developing and refining hypotheses about roles
- **Information Integration**: Combining disparate clues into coherent theories
- **Decision Under Uncertainty**: Making optimal choices with incomplete information
- **Adversarial Reasoning**: Understanding others' strategic motivations
- **Pattern Recognition**: Identifying consistent behavior across responses
- **Predictive Modeling**: Anticipating responses based on role hypotheses

### 3. Social Deception Capabilities

- **Lie Crafting**: For lying roles, producing believable false statements
- **Truth Masking**: Presenting true information in misleading ways
- **Misdirection**: Diverting attention from revealing information
- **Strategic Disclosure**: Selecting what information to reveal
- **False Confidence**: Projecting certainty despite internal uncertainty
- **Consistency Maintenance**: Ensuring responses don't contradict previous statements
- **Deceptive Coordination**: Aligning deception with team members

### 4. Political Reasoning

- **Coalition Dynamics**: Understanding team-based incentive structures
- **Role-based Behavior Modeling**: Predicting actions based on role constraints
- **Strategic Communication**: Using questions to achieve specific information goals
- **Power Dynamics**: Understanding the asymmetric influence of different roles
- **Competing Narratives**: Evaluating alternative explanations for observed behavior
- **Incentive Analysis**: Identifying motivations behind different statements
- **Strategic Adaptation**: Adjusting theories based on new evidence

## Research Applications

The TofuKingdom environment serves as an experimental platform for research into:

### Political Agent Development

- **Deception Detection**: Building systems that can identify false information
- **Strategic Questioning**: Designing optimal information-elicitation strategies
- **Coalition Behavior**: Modeling how aligned agents coordinate their statements
- **Competing Interests**: Understanding how conflicting goals shape communications
- **Truth Evaluation**: Developing frameworks for assessing statement veracity
- **Adversarial Information Environments**: Navigating spaces with intentional misinformation
- **Strategic Communication**: Balancing information extraction with revelation

### LLM Capability Assessment

- **Logical Reasoning**: Testing models' ability to solve complex logical puzzles
- **Contradiction Handling**: Assessing how models resolve inconsistent information
- **Strategic Planning**: Evaluating multi-step reasoning toward information goals
- **Theory of Mind**: Testing understanding of others' knowledge and intentions
- **Deceptive Capabilities**: Measuring both generation and detection of deception
- **Role-Constrained Behavior**: Assessing adherence to defined behavioral rules
- **Epistemic Uncertainty**: Handling multiple possible world-states simultaneously

### Game Theory Applications

- **Information Asymmetry**: Studying decision-making with unequal information access
- **Signaling Games**: Analyzing credible and non-credible information transmission
- **Coalition Formation**: Examining how shared incentives create aligned behaviors
- **Strategic Deception**: Exploring optimal deception strategies
- **Bayesian Updating**: Studying belief revision with unreliable information
- **Coordination Problems**: Investigating team-based information sharing
- **Mixed-motive Interactions**: Analyzing scenarios with partially aligned incentives

## Technical Implementation

### File Structure

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

### Metrics Implementation

The metrics system integrates with the PolitAgent framework and provides:

1. **Question Analysis**: Evaluation of Prince's questioning strategy and effectiveness
2. **Truth/Lie Tracking**: Measurement of deception generation and detection
3. **Role Performance**: Assessment of how well each role follows their behavior rules
4. **Team Dynamics**: Analysis of coordination within and between teams
5. **Outcome Analysis**: Detailed breakdown of game results and contributing factors
6. **LLM-as-Judge**: Optional evaluation using an external model to assess strategic play

### Game Flow

1. **Initialization**:
   - Randomly assign roles to players
   - Set up agents with appropriate models for each team
   - Configure truth behavior rules for each role
   - Record role assignments for metrics tracking

2. **Question Rounds**:
   - Prince selects and asks one question to each player
   - Each player responds according to their role's truth behavior
   - Responses are analyzed and recorded in metrics
   - Prince updates internal beliefs based on answers

3. **Final Question**:
   - Prince selects one player for a decisive final question
   - The chosen player responds according to their role
   - Prince makes final information update

4. **Decision Phase**:
   - Prince makes final determination of Princess identity
   - System determines winning team based on the guess
   - Comprehensive metrics calculation
   - Optional LLM evaluation of strategic performance

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

### Full Benchmark Mode

```bash
python -m core.benchmark --full_benchmark --games tofukingdom --models openai --specific_model gpt-4
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
| `--use_llm_evaluation` | Enable LLM-as-judge for game evaluation | `False` |

## Political Agent Implications

The TofuKingdom environment offers valuable insights for political agent development:

### 1. Information Reliability Assessment

- **Source Credibility**: How to evaluate information based on source reputation and incentives
- **Consistency Analysis**: Detecting contradictions across multiple statements
- **Pattern Recognition**: Identifying behavioral patterns that reveal underlying motives
- **Strategic Questioning**: Designing questions that maximize information extraction
- **Cross-Verification**: Using multiple information sources to triangulate truth

### 2. Strategic Deception

- **Plausible Deniability**: Creating statements that appear truthful while being false
- **Partial Truth Tactics**: Mixing accurate and inaccurate information strategically
- **Coordinated Misinformation**: Aligning deceptive statements across multiple actors
- **Deception Detection**: Identifying when others are providing false information
- **Reputation Management**: Maintaining credibility while pursuing strategic goals

### 3. Coalition Dynamics

- **Team-based Incentives**: How shared objectives shape information sharing
- **Competitive Cooperation**: Balancing cooperation within teams against competition between teams
- **Information Compartmentalization**: Strategic sharing and withholding within coalitions
- **Coordinated Narrative Building**: Creating consistent alternate realities through aligned statements
- **Mixed Loyalty Challenges**: Managing complex incentive structures in multi-polar systems

### 4. Constrained Strategic Action

- **Rule-Bound Behavior**: Operating effectively within strict behavioral constraints
- **Format Optimization**: Maximizing strategic impact within limited action formats
- **Sequential Decision Optimization**: Making the most of limited action opportunities
- **Decision Under Uncertainty**: Making optimal choices with incomplete information
- **Final Decision Quality**: Converting partial information into effective conclusions

Through systematic experimentation with the TofuKingdom environment, researchers can develop more sophisticated political agents capable of navigating complex information environments with strategic deception, competing interests, and coalition dynamics that mirror real-world political systems. 