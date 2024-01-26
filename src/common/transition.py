class Transition:

    def __init__(self, t: int, state: dict, action_id: int, action: dict, reward: float, terminal_state: str,
                 timestamp: float):
        self.t: int = t
        self.state: dict = state
        self.action_id: int = action_id
        self.action: dict = action
        self.reward: float = reward
        self.terminal_state: str = terminal_state
        self.timestamp: float = timestamp
