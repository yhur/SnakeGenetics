"""All tunable settings for the network, the genetic algorithm and the game."""
from dataclasses import dataclass, field, asdict
import json
import os

# The settings the dashboard is using (written on Apply and on Load). Both modes start from them.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'settings.json')


# Settings that decide the shape of the brain. A saved snake's genome only fits the brain
# it was trained with, so these always come from the checkpoint when snakes are loaded.
BRAIN_SETTINGS = ['hidden_layers', 'hidden_activation', 'output_activation',
                  'input_type', 'vision_directions', 'apple_self_vision']


@dataclass
class Config:
    # ---- Game -------------------------------------------------------------
    grid_x: int = 20
    grid_y: int = 20

    # ---- Neural network ----------------------------------------------------
    hidden_layers: list = field(default_factory=lambda: [20, 12])   # neurons per hidden layer
    hidden_activation: str = 'relu'          # relu | leaky_relu | tanh | sigmoid
    output_activation: str = 'sigmoid'       # sigmoid | softmax | linear
    input_type: str = 'vision'               # vision: rays (see snake_ai.py) | simple: 14 yes/no facts
    vision_directions: int = 8               # vision: 8 or 4 rays
    apple_self_vision: str = 'binary'        # vision: binary | distance

    # ---- Genetic algorithm -------------------------------------------------
    num_parents: int = 100                   # mu: survivors kept each generation
    num_offspring: int = 400                 # lambda: children bred each generation
    selection_type: str = 'plus'             # plus (mu+lambda) | comma (mu,lambda)
    parent_selection: str = 'tournament'     # tournament | roulette
    tournament_size: int = 5

    crossover_rate: float = 1.0              # probability that two parents are crossed (else cloned)
    sbx_ratio: float = 0.5                   # share of crossovers done with SBX; the rest use the point-crossover below
    sbx_eta: float = 100.0                   # SBX distribution index
    point_crossover: str = 'single'          # single | two | uniform
    cross_point_mode: str = 'random'         # random | fixed
    cross_point: float = 0.5                 # fixed cut position as a fraction of each layer's genes
    cross_point2: float = 0.75               # second cut for two-point crossover (fixed mode)

    mutation_rate: float = 0.05              # probability that each gene mutates
    mutation_rate_type: str = 'static'       # static | decaying
    gaussian_ratio: float = 1.0              # share of mutations that are Gaussian; rest are uniform
    gaussian_sigma: float = 0.2
    uniform_range: float = 1.0               # uniform mutation replaces a gene with U(-r, r)
    lifespan: int = 0                        # generations an individual may survive (0 = infinite)

    seed: int = 0                            # 0 = random

    # ---------------------------------------------------------------------
    @property
    def num_inputs(self):
        if self.input_type == 'simple':
            return 14                                # wall F/R/L + moving R/D/L/U + body F/R/L + apple L/R/U/D
        return self.vision_directions * 3 + 4 + 4   # rays x (wall, apple, self) + head dir + tail dir

    @property
    def num_outputs(self):
        return 3                                     # [forward, right, left] -- the game's action encoding

    @property
    def layer_sizes(self):
        return [self.num_inputs] + list(self.hidden_layers) + [self.num_outputs]

    @property
    def population_size(self):
        return self.num_parents + self.num_offspring if self.selection_type == 'plus' else self.num_offspring

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_last(cls):
        """The settings last applied in the dashboard, or the defaults if there are none yet."""
        if os.path.exists(SETTINGS_FILE):
            return cls.load(SETTINGS_FILE)
        return cls()

    def same_brain_as(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in BRAIN_SETTINGS)

    def with_brain_of(self, other):
        """A copy of these settings, but with the brain settings taken from `other`."""
        d = self.to_dict()
        for name in BRAIN_SETTINGS:
            d[name] = getattr(other, name)
        return Config.from_dict(d)
