# Enhanced Diplomacy Metrics System

Comprehensive evaluation framework for LLM performance in Diplomacy game environments, featuring detailed inference analysis, strategic evaluation, and human-readable reporting.

## Overview

The enhanced Diplomacy metrics system provides a multi-dimensional evaluation framework that goes beyond simple win/loss statistics to assess:

- **Model Inference Performance**: Detailed analysis of how well the model handles different types of reasoning tasks
- **Strategic Capabilities**: Long-term planning, adaptability, and resource management
- **Tactical Execution**: Order quality, coordination, and timing
- **Diplomatic Skills**: Negotiation effectiveness, alliance building, and communication quality
- **Behavioral Patterns**: Consistency, risk-taking, and adaptation to game flow

## Key Features

### 🔍 Comprehensive Inference Metrics
- **Total Inference Tracking**: Complete count of all model inferences across game phases
- **Quality Assessment**: Evaluation of response quality using both rule-based and contextual metrics
- **Error Analysis**: Detection and categorization of model errors and invalid outputs
- **Consistency Analysis**: Measurement of decision consistency across game rounds
- **Context Utilization**: Assessment of how well the model uses available game state information

### 🎯 Strategic Performance Evaluation
- **Win Rate Analysis**: Detailed breakdown by power and game conditions
- **Territorial Management**: Analysis of expansion patterns and defensive strategies
- **Supply Center Control**: Tracking of resource acquisition and retention
- **Strategic Positioning**: Evaluation of long-term positional advantages

### ⚔️ Tactical Assessment
- **Order Quality**: Analysis of command syntax, strategic value, and execution timing
- **Attack/Defense Coordination**: Evaluation of tactical coordination and support usage
- **Unit Efficiency**: Assessment of how effectively units are utilized
- **Combat Effectiveness**: Success rates in conflicts and territorial disputes

### 🤝 Diplomatic Analysis
- **Negotiation Effectiveness**: Quality and success rate of diplomatic communications
- **Alliance Formation**: Ability to build and maintain strategic partnerships
- **Trust Management**: Analysis of promise-keeping and betrayal detection
- **Communication Quality**: Linguistic and strategic quality of diplomatic messages

### 🤖 LLM-as-Judge Evaluation
- **Multi-dimensional Scoring**: Strategic, diplomatic, tactical, and overall performance scores (1-10)
- **Contextual Analysis**: Deep evaluation of decision quality given game context
- **Comparative Assessment**: Relative performance evaluation across different powers
- **Qualitative Insights**: Natural language explanations of strengths and weaknesses

### 📊 Comprehensive Reporting
- **Markdown Reports**: Human-readable analysis with clear visualizations
- **JSON Data Export**: Machine-readable data for further analysis
- **Comparative Analysis**: Side-by-side model comparison capabilities
- **Executive Summaries**: High-level insights for quick assessment

## Usage

### Basic Usage

```python
from metrics.diplomacy_metrics import DiplomacyMetrics
from llm.openai_chat import OpenAIChat
from llm.config import LLMConfig

# Initialize metrics with optional LLM evaluator
config = LLMConfig()
evaluator = OpenAIChat(config, model_name="gpt-4o")
metrics = DiplomacyMetrics(model=evaluator)

# Calculate comprehensive metrics
results = metrics.calculate_metrics("path/to/game/results")

# Generate detailed report
metrics.save_detailed_report("diplomacy_analysis_report")
```

### Model Comparison

```python
# Compare multiple models
model_results = {
    "GPT-4": "results/gpt4_diplomacy",
    "Claude-3": "results/claude3_diplomacy", 
    "Llama-3": "results/llama3_diplomacy"
}

for model_name, results_dir in model_results.items():
    metrics = DiplomacyMetrics(model=evaluator)
    results = metrics.calculate_metrics(results_dir)
    metrics.save_detailed_report(f"{model_name}_analysis")
```

### Advanced Analysis

```python
# Access specific metric categories
model_performance = results["model_performance"]
strategic_metrics = results["strategic_metrics"]
llm_evaluation = results["llm_evaluation"]

# Detailed inference analysis
print(f"Total Inferences: {model_performance['total_inferences']}")
print(f"Error Rate: {model_performance['total_errors'] / model_performance['total_inferences']:.2%}")
print(f"Average Response Quality: {np.mean(list(model_performance['response_quality'].values())):.3f}")
```

## Metric Categories

### 1. Model Inference Performance

| Metric | Description | Range |
|--------|-------------|-------|
| `total_inferences` | Total number of model calls | 0+ |
| `negotiation_inferences` | Inferences for diplomatic communication | 0+ |
| `action_inferences` | Inferences for military orders | 0+ |
| `strategic_inferences` | Inferences for strategic decisions | 0+ |
| `response_quality` | Quality of model outputs | 0.0-1.0 |
| `decision_consistency` | Consistency across rounds | 0.0-1.0 |
| `context_utilization` | Use of available context | 0.0-1.0 |
| `error_rate` | Rate of invalid/malformed outputs | 0.0-1.0 |

### 2. Strategic Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `win_rate_by_power` | Win percentage for each power | 0.0-1.0 |
| `supply_centers_by_power` | Average supply centers controlled | 0.0-34.0 |
| `survival_rate_by_power` | Percentage of games survived | 0.0-1.0 |
| `territorial_expansion` | Net territorial growth | -∞ to +∞ |
| `strategic_positioning` | Long-term positioning score | 0.0-1.0 |

### 3. Tactical Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `attack_success_rate` | Success rate of offensive actions | 0.0-1.0 |
| `defense_success_rate` | Success rate of defensive actions | 0.0-1.0 |
| `support_coordination` | Use of support orders | 0.0-1.0 |
| `unit_efficiency` | Utilization of available units | 0.0-1.0 |
| `order_complexity` | Complexity of issued orders | 0.0-1.0 |

### 4. Diplomatic Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `alliance_effectiveness` | Success of alliance formation | 0.0-1.0 |
| `negotiation_success_rate` | Success of diplomatic proposals | 0.0-1.0 |
| `negotiation_honesty` | Consistency between words and actions | 0.0-1.0 |
| `communication_quality` | Quality of diplomatic messages | 0.0-1.0 |
| `alliance_formation` | Ability to form stable alliances | 0.0-1.0 |

### 5. LLM Judge Evaluation

| Metric | Description | Range |
|--------|-------------|-------|
| `strategic_avg` | Strategic performance scores | 1.0-10.0 |
| `diplomatic_avg` | Diplomatic performance scores | 1.0-10.0 |
| `tactical_avg` | Tactical performance scores | 1.0-10.0 |
| `overall_avg` | Overall performance scores | 1.0-10.0 |

## Report Structure

### Executive Summary
- Total games analyzed
- Key performance indicators
- Best/worst performing powers
- Overall error rates and quality metrics

### Model Performance Analysis
- Detailed inference breakdown
- Quality metrics across all categories
- Error analysis with recommendations
- Performance insights and patterns

### Strategic Analysis
- Win rate analysis by power
- Territorial control patterns
- Resource management effectiveness
- Long-term strategic positioning

### Tactical Analysis
- Combat effectiveness metrics
- Order quality assessment
- Coordination and support usage
- Unit utilization efficiency

### Diplomatic Analysis
- Negotiation effectiveness
- Alliance formation and maintenance
- Communication quality assessment
- Trust and reputation management

### LLM Judge Evaluation
- Expert-level assessment scores
- Contextual performance analysis
- Qualitative insights and explanations
- Comparative power rankings

### Behavioral Analysis
- Aggression and cooperation patterns
- Risk-taking tendencies
- Adaptability to game flow
- Consistency across different scenarios

### Recommendations
- Specific improvement suggestions
- Areas for model enhancement
- Strategic and tactical insights
- Training recommendations

## Configuration

### Game Data Format

The metrics system expects game data in the following JSON format:

```json
{
  "rounds_played": 8,
  "winner": "FRANCE",
  "game_time": 1847.3,
  "supply_centers": {
    "FRANCE": 18,
    "GERMANY": 8,
    ...
  },
  "strategic_decisions": {
    "FRANCE": "Focus on central Europe expansion",
    ...
  },
  "rounds_data": [
    {
      "round": 1,
      "year": 1901,
      "phase": "Spring",
      "territories_before": {...},
      "territories_after": {...},
      "negotiations": {...},
      "orders": {...},
      "attacks_received": {...}
    },
    ...
  ]
}
```

### LLM Evaluator Configuration

```python
# Configure LLM evaluator for advanced analysis
from llm.openai_chat import OpenAIChat
from llm.config import LLMConfig

config = LLMConfig()
evaluator = OpenAIChat(
    config, 
    model_name="gpt-4o",  # Use GPT-4 for best evaluation quality
    temperature=0.1       # Low temperature for consistent evaluation
)

metrics = DiplomacyMetrics(model=evaluator)
```

## Best Practices

### 1. Data Collection
- Ensure complete game logs with all required fields
- Include detailed negotiation transcripts for diplomatic analysis
- Capture all model inference attempts, including errors
- Record timing information for performance analysis

### 2. Evaluation Setup
- Use a high-quality LLM (GPT-4, Claude-3 Opus) for LLM-as-judge evaluation
- Set low temperature for consistent evaluation
- Validate evaluation prompts with domain experts
- Run multiple evaluation rounds for statistical significance

### 3. Analysis and Interpretation
- Consider multiple metrics together for comprehensive assessment
- Account for power-specific advantages and challenges
- Compare against baseline human or random performance
- Use comparative analysis for model selection decisions

### 4. Reporting and Communication
- Generate both technical and executive-level reports
- Include confidence intervals and statistical significance
- Provide actionable recommendations for improvement
- Visualize trends and patterns clearly

## Integration with PolitAgent

The enhanced metrics system integrates seamlessly with the PolitAgent framework:

```python
# In your game runner script
from environments.diplomacy_game.game import DiplomacyGame
from metrics.diplomacy_metrics import DiplomacyMetrics

# Run games
game = DiplomacyGame(args, model)
results = game.game_loop()

# Calculate comprehensive metrics
metrics = DiplomacyMetrics(model=evaluator_model)
analysis = metrics.calculate_metrics("results/")
metrics.save_detailed_report("analysis_report")
```

## Example Output

### Console Summary
```
============================================================
DIPLOMACY MODEL PERFORMANCE SUMMARY
============================================================

Games Analyzed: 10

MODEL INFERENCE METRICS:
  Total Inferences: 2,847
  Negotiation Inferences: 1,203
  Action Inferences: 980
  Strategic Inferences: 664
  Total Errors: 23
  Average Response Quality: 0.847
  Average Decision Consistency: 0.723
  Average Context Utilization: 0.891

STRATEGIC PERFORMANCE:
  Best Power: FRANCE (45.00% win rate)
  Worst Power: TURKEY (10.00% win rate)

LLM JUDGE EVALUATION:
  Best Overall Performance: FRANCE (8.2/10)

GAME OUTCOME ANALYSIS:
  Average Game Length: 12.3 rounds
  Decisive Victories: 7
  Draws: 3
  Fastest Victory: 8 rounds
============================================================
```

### Generated Reports
- `analysis_report.md`: Human-readable markdown report
- `analysis_report.json`: Machine-readable data for further analysis

## Future Enhancements

- **Real-time Analysis**: Live performance monitoring during games
- **Advanced Visualizations**: Interactive charts and graphs
- **Comparative Benchmarking**: Performance against human players
- **Meta-learning Analysis**: Pattern recognition across multiple games
- **Automated Insights**: AI-generated observations and recommendations

## Troubleshooting

### Common Issues

1. **Missing Game Data Fields**
   - Ensure all required fields are present in game logs
   - Check field names match expected format

2. **LLM Evaluation Errors**
   - Verify API credentials and model availability
   - Check rate limits and adjust accordingly
   - Validate prompt templates for clarity

3. **Performance Issues**
   - Consider sampling for large datasets
   - Use parallel processing for multiple games
   - Optimize data loading and processing

### Support

For questions, issues, or contributions to the Diplomacy metrics system:
- Create issues in the repository
- Review existing documentation and examples
- Contact the development team for advanced use cases 