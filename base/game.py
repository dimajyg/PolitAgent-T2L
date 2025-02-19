from abc import ABC, abstractmethod
import json
import os

class Game(ABC):
    def __init__(self, args):
        self.args = args
        self.debug = args.debug if hasattr(args, 'debug') else False
        self.players = []
        self.agents = []
        self.name2agent = {}
        self.public_messages = []
        self.game_round = 0

    @abstractmethod
    def init_game(self):
        """Initialize game state, agents, and settings"""
        pass

    @abstractmethod
    def game_loop(self):
        """Main game loop implementation"""
        pass

    def update_history(self, message, sender_name):
        """Update message history for all agents except the sender"""
        for agent in self.agents:
            if agent.player_name != sender_name:
                agent.private_history.append(message)
        self.public_messages.append(message)

    def log_message(self, file, message, cot=None):
        """Log message and chain of thought to file"""
        file.write(message + "\n")
        if cot:
            file.write(json.dumps(cot) + "\n")
        if self.debug:
            print(message)
            if cot:
                print(json.dumps(cot))
            print()

    def create_log_dir(self, base_dir):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir

    def run(self, log_file):
        """Run a complete game session"""
        try:
            self.init_game()
            return self.game_loop(log_file)
        except Exception as e:
            if self.debug:
                print(f"Error: {str(e)}")
            return {"error": str(e)}