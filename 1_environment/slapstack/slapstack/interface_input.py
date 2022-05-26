from slapstack.interface_templates import SimulationParameters


class Input:
    def __init__(
            self, environment_parameters: SimulationParameters, seed: int = 1):
        self.params = environment_parameters
        self.seed = seed
