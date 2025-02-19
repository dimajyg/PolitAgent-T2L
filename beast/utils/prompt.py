def get_role_prompt(player_name, wealth):
    return f"""You are {player_name} in a strategic wealth game.\n
            Your current wealth is {wealth}.\n
            Your goal is to accumulate as much wealth as possible.\n
            You can have conversations with other players, make money offers, and vote for players.\n
            The player with the most votes each round wins 250,000 but the game will end for him, he would not contend in voting anymore.\n
            Only first 5 players can get elimination bonus, so you will want to be eliminated at first 5 rounds.\n
            Eliminated player can't share his money anyway, all the wealth is satying with him.\n
            The game continues until 5 players are eliminated.\n
            Be strategic in your conversations and voting decisions.\n"""

def get_current_wealth_prompt(wealth: dict[str, int]) -> str:
    return ''.join(f"{player} has {amount}\n" for player, amount in wealth.items())

def get_choose_conv_prompt(players):
    return f"""Choose 2-3 players from {players} to have conversations with.
            Consider the following strategic factors when making your choice:
            - Players who might be willing to form alliances
            - Players who could be convinced to vote for you
            - Players who might share valuable information about others' strategies
            - Players whose wealth status makes them potential allies or targets
            
            Make your choices carefully as these conversations will influence voting outcomes.
            Remember that building trust with some players while managing potential threats 
            from others is key to your survival and success in the game."""

def get_conv_prompt(opp_name):
    return f"""You are starting a conversation with {opp_name}.
            This is your opportunity to:
            - Build trust and potential alliances
            - Make him vote for you
            - Negotiate wealth transfers that benefit both parties
            - Gather information about other players' strategies
            - Discuss voting strategies and potential collaborations
            
            Remember to be diplomatic but strategic in your interactions.
            Your ultimate goal is to accumulate wealth while ensuring your survival in the game."""

def get_voting_prompt(voting_results: dict[str, int]):
    return ''.join(f"{player} has {votes}\n" for player, votes in voting_results.items())
    