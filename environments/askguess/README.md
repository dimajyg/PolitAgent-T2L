# AskGuess Game Environment - Strategic Information Elicitation Benchmark

AskGuess is a sophisticated question-answering game that evaluates language models' abilities in strategic information gathering, hypothesis formation, and adaptive reasoning. This environment simulates critical aspects of investigative dialogue, providing a controlled testing ground for measuring how effectively LLMs can navigate unknown information spaces through strategic questioning.

## Strategic Testing Framework

The AskGuess environment rigorously assesses language models' capabilities in:

- **Strategic Information Elicitation**: Formulating questions that maximize information gain
- **Hypothesis Formation and Testing**: Developing and refining theories about the target concept
- **Adaptive Questioning**: Adjusting inquiry strategy based on accumulated information
- **Constraint Satisfaction**: Narrowing possibilities through systematic elimination
- **Question Sequencing**: Organizing queries to build upon previous answers
- **Deductive Reasoning**: Drawing logical conclusions from partial information
- **Semantic Understanding**: Recognizing conceptual relationships and category structures

## Game Architecture

### Core Game Mechanics

AskGuess implements a dynamic question-answer interaction framework:

#### 1. Information Asymmetry System
- One agent (AnswerAgent) knows a hidden target word
- Another agent (QuestionAgent) must discover this word through questioning
- Success requires efficiently narrowing the potential solution space
- Game difficulty can be adjusted through constraints on question types

#### 2. Game Modes

##### Easy Mode
- Initial context: Answerer provides a brief description of the target concept
- Question format: Open-ended questions allowed
- Answer format: Informative responses permitted
- Success metric: Discover the word in minimum questions

##### Hard Mode
- Initial context: No description provided
- Question format: Only yes/no questions allowed
- Answer format: Restricted to "yes", "no", or "gameover"
- Success metric: Discover the word within the question limit

#### 3. Strategic Constraints
- Fixed maximum number of questions (typically 10 or 20)
- Answering agent cannot directly reveal the target word
- Questions must build on previously acquired information
- The system monitors for direct word mentions (cheating prevention)

### Advanced Features

- **Thinking Chain Capture**: Recording internal reasoning processes during question formulation
- **Convergence Tracking**: Measuring progress toward the solution over successive questions
- **Information Gain Analysis**: Evaluating how much each question narrows the solution space
- **Adaptive Difficulty**: Optional dynamic adjustment of question constraints
- **Knowledge Domain Variation**: Testing across different concept categories and complexity levels
- **Multi-round Strategic Analysis**: Examining question sequences across complete game trajectories

## Comprehensive Metrics System

AskGuess employs a sophisticated metrics framework to evaluate agent performance:

### 1. Strategic Questioning Metrics

- **Information Gain Per Question**: Measures how effectively each question reduces uncertainty
- **Question Efficiency**: Evaluates the ratio of useful information to question complexity
- **Strategic Adaptation**: Tracks how questions evolve based on previous answers
- **Question Diversity**: Assesses the range of questioning approaches used
- **Question Sequencing Logic**: Measures coherence in the progression of questions
- **Category Exploration**: Analyzes systematic coverage of potential concept categories
- **Feature Targeting**: Evaluates focus on distinctive features of potential targets

### 2. Deductive Reasoning Metrics

- **Hypothesis Evolution**: Tracks refinement of target hypotheses across rounds
- **Logical Consistency**: Measures adherence to logical constraints from previous answers
- **Convergence Speed**: Evaluates how quickly the agent narrows toward the correct answer
- **Elimination Efficiency**: Assesses systematic ruling out of possibilities
- **Inference Quality**: Evaluates correctness of deductions from available information
- **Conjecture Testing**: Measures strategic testing of tentative hypotheses
- **Decision Tree Optimization**: Analyzes efficiency of the question decision tree

### 3. Cognitive Process Metrics

- **Thinking Chain Analysis**: Evaluates the reasoning process behind question selection
- **Memory Utilization**: Assesses how effectively previous answers inform new questions
- **Abstraction Capability**: Measures ability to reason at different conceptual levels
- **Constraint Satisfaction**: Evaluates adherence to established constraints
- **Conceptual Mapping**: Assesses construction of semantic relationships
- **Self-correction**: Measures adaptation when pursuing incorrect hypotheses
- **Metacognitive Awareness**: Evaluates self-monitoring of information needs

### 4. Performance Outcome Metrics

- **Success Rate**: Percentage of games where the word is correctly guessed
- **Average Questions to Solution**: Number of questions needed to find the answer
- **Time Efficiency**: How quickly the correct answer is reached
- **Error Analysis**: Types and patterns of reasoning failures
- **Concept Difficulty Correlation**: Performance relative to concept complexity
- **Strategic Consistency**: Reliability of questioning strategy across different targets
- **Domain Adaptation**: Performance variation across different knowledge domains

### 5. LLM Evaluation Metrics

- **Question Quality**: Expert LLM assessment of question strategic value
- **Information Gain Efficiency**: Evaluation of information extracted per question
- **Reasoning Process**: Analysis of the thinking chain behind questions
- **Strategy Optimization**: Assessment of overall questioning strategy effectiveness
- **Game-level Performance**: Comprehensive evaluation of complete game trajectories
- **Counterfactual Analysis**: Identification of missed strategic opportunities
- **Improvement Recommendations**: Targeted suggestions for strategy enhancement

## LLM Capability Assessment

Through the AskGuess benchmark, we can assess several critical capabilities of language models:

### 1. Strategic Reasoning

- **Optimal Information Seeking**: Ability to maximize information gain with each question
- **Search Space Reduction**: Efficiently narrowing possibilities through strategic questions
- **Decision Tree Navigation**: Optimally traversing the tree of possible concepts
- **Trade-off Analysis**: Balancing specificity and breadth in questioning
- **Strategy Adjustment**: Adapting approach based on new information
- **Efficiency Optimization**: Minimizing questions needed to identify the target
- **Uncertainty Management**: Operating effectively with incomplete information

### 2. Cognitive Modeling

- **Knowledge Representation**: How models organize conceptual information
- **Semantic Network Navigation**: Traversal of conceptual relationships
- **Categorical Reasoning**: Using hierarchical category structures
- **Feature-based Reasoning**: Focusing on distinctive attributes
- **Abductive Reasoning**: Generating best explanations for observed clues
- **Analogical Thinking**: Drawing parallels to related concepts
- **Bayesian Updating**: Revising beliefs based on new evidence

### 3. Meta-Strategic Thinking

- **Question Planning**: Developing multi-step questioning strategies
- **Information Value Assessment**: Prioritizing different types of information
- **Strategy Selection**: Choosing appropriate questioning approaches
- **Counterfactual Analysis**: Considering alternative question paths
- **Time Horizon Management**: Balancing immediate vs. long-term information needs
- **Exploitation vs. Exploration**: Trading off between narrowing in and exploring
- **Failure Recovery**: Adapting after receiving unexpected answers

### 4. Language Understanding

- **Semantic Precision**: Understanding nuanced meanings in answers
- **Implicit Information Detection**: Drawing inferences from indirect answers
- **Contextual Integration**: Connecting new information with established context
- **Ambiguity Resolution**: Clarifying vague or uncertain responses
- **Abstract Concept Manipulation**: Working with non-concrete conceptual targets
- **Definition Extraction**: Constructing working definitions from partial information
- **Semantic Boundary Testing**: Probing conceptual boundaries and distinctions

## Research Applications

The AskGuess environment serves as an experimental platform for research into:

### Political Agent Development

- **Strategic Information Gathering**: Testing how agents extract critical information
- **Investigative Dialogue**: Modeling effective information-seeking discourse
- **Hypothesis Formation**: Developing and refining theories about unknown situations
- **Strategic Communication**: Balancing direct and indirect questioning approaches
- **Adaptive Inquiry**: Responding to new information by redirecting investigation
- **Knowledge Elicitation**: Extracting information from information-holding agents
- **Efficient Investigation**: Optimizing question sequences under time constraints

### LLM Capability Assessment

- **Question Generation**: Testing ability to formulate informative questions
- **Semantic Reasoning**: Evaluating understanding of conceptual relationships
- **Logical Inference**: Measuring deductive reasoning from partial information
- **Memory Integration**: Assessing cumulative information processing
- **Conceptual Modeling**: Testing construction of conceptual frameworks
- **Strategy Optimization**: Measuring systematic improvement in questioning
- **Failure Mode Analysis**: Identifying patterns in reasoning breakdowns

### Dialogue System Applications

- **Interactive FAQ Systems**: Improving question interpretation and information delivery
- **Investigative Chatbots**: Enhancing systematic information gathering
- **Expert Systems**: Developing efficient knowledge elicitation systems
- **Educational Tutoring**: Creating adaptive questioning for learning assessment
- **Customer Support**: Optimizing problem diagnosis through strategic questioning
- **Interview Systems**: Improving information extraction through question sequencing
- **Decision Support**: Enhancing decision-making through structured information gathering

## Technical Implementation

### File Structure

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

### Metrics Implementation

The metrics system integrates with the PolitAgent framework and provides:

1. **Question-level Analysis**: Evaluation of individual question quality and information gain
2. **Game-level Analysis**: Assessment of overall strategy and performance across games
3. **Thinking Chain Capture**: Recording and analysis of reasoning processes
4. **Information Gain Modeling**: Mathematical modeling of solution space reduction
5. **Comparative Analysis**: Cross-model performance comparison under identical conditions
6. **LLM-as-Judge**: Optional evaluation using an external model to assess questioning quality

### Game Flow

1. **Initialization**:
   - Select target word from knowledge domain
   - Configure game parameters (mode, round limit)
   - Initialize agents with appropriate prompts

2. **Description Phase** (Easy mode only):
   - AnswerAgent provides initial description of the concept
   - System verifies the description doesn't reveal the word
   - QuestionAgent receives the description as context

3. **Question-Answer Loop**:
   - QuestionAgent formulates a strategic question
   - Internal thinking process is recorded for analysis
   - AnswerAgent provides appropriate response
   - System checks for rule compliance
   - Question and answer are added to game history

4. **Guess Evaluation**:
   - When QuestionAgent makes a guess, it's evaluated against the target
   - If correct, game ends successfully
   - If incorrect, questioning continues (unless max rounds reached)

5. **Game Resolution**:
   - Success: Target word correctly identified
   - Timeout: Maximum questions reached
   - Comprehensive metrics calculation

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

### Full Benchmark Mode

```bash
python -m core.benchmark --full_benchmark --games askguess --models openai --specific_model gpt-4
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--label_path` | Path to JSON file with words to guess | `environments/askguess/test_labels.json` |
| `--mode` | Game mode ("easy" or "hard") | `hard` |
| `--max_rounds` | Maximum number of question rounds | `10` |
| `--model_name` | LLM provider to use | `openai` |
| `--max_phrases` | Limit number of words to test | All words in the file |
| `--use_llm_evaluation` | Enable LLM-as-judge for question quality assessment | `False` |

## Political Agent Implications

The AskGuess environment offers valuable insights for political agent development:

### 1. Information-Gathering Strategies

- **Strategic Question Formulation**: How to maximize information gain while minimizing questions
- **Ambiguity Management**: Techniques for resolving unclear or incomplete information
- **Question Sequence Optimization**: Building effective chains of questions
- **Adaptive Investigation**: Modifying inquiry approach based on emerging information
- **Domain Exploration**: Systematically exploring unknown information spaces

### 2. Hypothesis Management

- **Multiple Hypothesis Tracking**: Maintaining and updating multiple possible explanations
- **Evidence Integration**: Incorporating new information into existing hypotheses
- **Contradictory Evidence Handling**: Resolving conflicting information
- **Confidence Calibration**: Appropriately assessing certainty levels for hypotheses
- **Hypothesis Testing Efficiency**: Designing optimal tests for competing theories

### 3. Strategic Deduction

- **Logical Constraint Satisfaction**: Finding solutions that satisfy all established facts
- **Elimination Reasoning**: Systematically ruling out possibilities
- **Inference Chain Construction**: Building robust logical pathways to conclusions
- **Knowledge Gap Identification**: Recognizing what critical information is missing
- **Conceptual Boundary Testing**: Determining precise definitions and limitations

### 4. Knowledge Elicitation

- **Information Source Management**: Extracting information effectively from informants
- **Question Framing Effects**: Impact of question phrasing on information quality
- **Leading vs. Open Question Trade-offs**: Balancing directed and exploratory questioning
- **Information Verification**: Cross-checking obtained information for consistency
- **Efficient Interviewing**: Minimizing questions while maximizing information extraction

Through systematic experimentation with the AskGuess environment, researchers can develop more sophisticated question-asking agents and improve strategic information gathering capabilities critical for political reasoning, investigative dialogue, and knowledge acquisition tasks. 