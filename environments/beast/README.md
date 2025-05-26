# Beast Game Environment - Enhanced Strategic Edition

Beast is an advanced strategic survival game where language model agents must navigate complex social dynamics, form secret alliances, manage hidden information, and make critical decisions under pressure to survive and accumulate wealth.

## Game Overview

Inspired by MrBeast's strategic challenge designs, this enhanced version features:
- **6-8 agents** compete in a high-stakes survival game with escalating pressure
- **Secret Information System** - Players have hidden roles, resources, and private information
- **Multi-Phase Challenges** - Each round has multiple decision points and strategic dilemmas
- **Alliance & Betrayal Mechanics** - Complex relationship systems with trust/betrayal tracking
- **Elimination Pressure** - Time limits and escalating stakes force quick decisions
- **Information Warfare** - Players can spread misinformation and gather intelligence
- **Strategic Resource Management** - Multiple currencies (wealth, influence, information, immunity)

## Enhanced Game Rules

### Initial Setup
1. **6-8 players** start with randomized hidden resources:
   - **Wealth**: 50,000-150,000 (hidden from others)
   - **Influence Points**: 0-3 (used for special actions)
   - **Secret Role**: Each player gets a hidden role with special abilities
   - **Private Information**: Each player knows 1-2 secrets about other players

### Secret Roles
- **The Insider**: Knows elimination target each round, can influence host decisions
- **The Banker**: Can secretly manipulate wealth transfers
- **The Spy**: Can discover other players' hidden information
- **The Manipulator**: Can spread false information effectively
- **The Guardian**: Can protect one player from elimination each round
- **The Saboteur**: Can block other players' special actions

### Game Phases (Each Round)

#### Phase 1: Intelligence Gathering (3 minutes)
- Players can choose 2 other players to investigate
- May discover: wealth levels, secret alliances, planned votes
- Can spread misinformation to others
- Risk: Being caught lying reduces trust meter

#### Phase 2: Secret Alliance Formation (5 minutes)
- Players can form private alliances (max 3 players per alliance)
- Alliance members share certain information
- Can create false alliances to deceive others
- Betraying an alliance has severe trust penalties

#### Phase 3: Strategic Challenge (Varies)
Different challenges each round:
- **Resource Auction**: Bid on immunity, information, or wealth
- **Prisoner's Dilemma**: Cooperate or defect for rewards/penalties
- **Trust Test**: Reveal true information or lie for strategic advantage
- **Sacrifice Choice**: Choose who loses resources for group benefit

#### Phase 4: Negotiation & Bribes (10 minutes)
- Open negotiations with time pressure
- Secret wealth transfers and promises
- Information trading and deals
- Last-minute alliance switching

#### Phase 5: Elimination Vote (2 minutes)
- Each player votes to eliminate one other player
- **Twist**: Random "Save" cards can protect players
- **Twist**: Some rounds have double eliminations
- **Twist**: Eliminated players can choose revenge targets

### Advanced Mechanics

#### Trust System
- Each player has a trust rating with every other player (hidden)
- Trust affects negotiation success rates and information sharing
- Betrayals and lies damage trust permanently
- High trust enables more effective alliances

#### Information Economy
Players can trade different types of information:
- **Public Info**: Known to everyone (current wealth rankings)
- **Semi-Secret**: Known to few (alliance memberships)
- **Secret**: Known to one (hidden roles, private resources)
- **False Info**: Deliberately planted misinformation

#### Pressure Escalation
- **Round 1-2**: Learning phase, low stakes
- **Round 3-4**: Medium pressure, first betrayals
- **Round 5-6**: High stakes, desperate alliances
- **Final Rounds**: Maximum pressure, all-or-nothing decisions

#### Win Conditions
Multiple victory paths:
1. **Wealth Victory**: Accumulate most total resources
2. **Survival Victory**: Be the last player standing
3. **Influence Victory**: Complete secret role objectives
4. **Alliance Victory**: Win as part of final alliance (shared victory)

## Enhanced Features

### Strategic Depth
- **Hidden Information**: Creates uncertainty and bluffing opportunities
- **Multiple Resources**: Players must balance wealth, influence, and relationships
- **Role Abilities**: Special powers create asymmetric gameplay
- **Time Pressure**: Forces quick decisions and prevents endless deliberation

### Diplomatic Complexity
- **Secret Communications**: Private alliance channels
- **Misinformation Campaigns**: Strategic lying and deception
- **Reputation Tracking**: Long-term consequences for actions
- **Betrayal Mechanics**: Risk/reward for breaking agreements

### Psychological Elements
- **Paranoia**: Hidden information creates suspicion
- **Pressure Cooker**: Time limits and elimination threats
- **Social Deduction**: Players must read intentions and spot lies
- **Risk Management**: Balancing aggressive and defensive strategies

## Technical Improvements

### Structured Decision Making
- **Pydantic Models**: All decisions use structured output for reliability
- **Decision Trees**: Complex multi-step decision processes
- **Time Management**: Strict timing controls prevent game stalls
- **Error Handling**: Fallback mechanisms for invalid decisions

### Advanced AI Behavior
- **Memory Systems**: Track relationships and past actions
- **Strategic Planning**: Multi-round planning capabilities
- **Personality Simulation**: Different AI behavioral patterns
- **Adaptive Strategies**: Learn from other players' patterns

### Metrics & Analysis
- **Trust Evolution**: Track relationship changes over time
- **Decision Quality**: Analyze strategic effectiveness
- **Information Flow**: Monitor misinformation spread
- **Alliance Dynamics**: Study cooperation patterns

## Game Balance

### Preventing Stagnation
- **Maximum Round Limit**: Game ends after 8 rounds regardless
- **Escalating Stakes**: Each round has higher penalties for inaction
- **Automatic Eliminations**: Random eliminations if no consensus
- **Time Pressure**: Strict time limits on all phases

### Encouraging Strategic Play
- **Multiple Victory Paths**: Different strategies can succeed
- **Hidden Information**: Rewards intelligence gathering and deduction
- **Alliance Benefits**: Cooperation provides real advantages
- **Betrayal Payoffs**: Well-timed betrayals are highly rewarded

### Testing AI Capabilities
- **Information Asymmetry**: Tests ability to reason with incomplete information
- **Social Dynamics**: Evaluates understanding of trust and relationships
- **Strategic Planning**: Requires multi-step thinking and adaptation
- **Deception & Detection**: Tests ability to lie convincingly and spot lies
- **Risk Assessment**: Evaluates decision-making under uncertainty

This enhanced Beast game provides a rigorous test of AI agents' strategic thinking, social intelligence, and ability to navigate complex multi-agent environments with hidden information and conflicting incentives.

## File Structure

```
beast/
├── game.py               # Main game implementation with game loop
├── agents/               # Agent implementations
│   ├── base_agent.py     # BeastAgent class with bargaining and voting logic
│   └── __init__.py       # Package exports
├── utils/                # Utility functions
│   ├── prompt.py         # Prompt template handling
│   └── utils.py          # Helper functions
├── prompts/              # Prompt text files
│   ├── role_prompt.txt               # Basic role description for players
│   ├── choose_conversation_prompt.txt # Instructions for choosing conversation partners
│   ├── conversation_prompt.txt       # Guidelines for negotiation conversations
│   ├── wealth_status_prompt.txt      # Format for displaying current wealth status
│   └── voting_results_prompt.txt     # Format for displaying voting results
└── __init__.py           # Package exports
```

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

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | LLM provider to use | `openai` |
| `--output_dir` | Directory for saving game results | `./results/beast` |
| `--max_rounds` | Number of elimination rounds | `5` |
| `--debug` | Enable verbose logging | `False` |

## Advanced Features

### Structured Output

Beast uses Pydantic models for structured LLM outputs to improve reliability:

- `BargainResponse`: Captures negotiation messages and offers
- `VoteResponse`: Ensures proper vote formatting

The system includes fallback mechanisms to handle potential output parsing errors.

### Game State Tracking

The game saves detailed state information at each round:
- Wealth of all players
- Elimination status
- Voting results
- Conversation history

These files are saved to the specified output directory for later analysis.

## Game Logic

1. **Initialization**:
   - Create 10 players with random initial wealth
   - Initialize agent conversation histories

2. **Conversation Stage**:
   - Update all agents with current wealth status
   - Agents choose opponents to converse with
   - Players engage in conversations and can make wealth transfers
   - Each conversation consists of multiple messages and potential offers

3. **Voting Stage**:
   - Each player votes for another player
   - Player with most votes is eliminated but receives a 250,000 wealth bonus
   - All players are informed about voting results

4. **Game End**:
   - After 5 players are eliminated, the game ends
   - Final rankings are determined by total wealth accumulated
   - Results are saved to JSON for analysis

## Game Results

The game returns a structured result dictionary with:
- `eliminated_players`: List of eliminated players and their final wealth
- `remaining_players`: List of surviving players and their wealth
- `total_rounds`: Number of rounds played
- `game`: Game identifier ("beast")

## Extending the Game

To customize the game behavior:

1. Modify prompt files in the `prompts/` directory to change agent behavior
2. Adjust conversation and voting logic in `game.py`
3. Add new agent strategies by extending the `BeastAgent` class
4. Implement additional bargaining mechanisms or voting strategies 