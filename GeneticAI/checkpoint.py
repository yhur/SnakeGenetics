"""Save and load training: settings, generation number, every genome, history and the best snake.

Both the dashboard and the headless trainer use these files, so you can train fast
in the background and then load the result in the dashboard to watch it.
"""
import json
import os
import numpy as np

from .config import Config
from .genetic import GeneticAlgorithm, Individual


def with_input_type(path, cfg):
    """Add the input type to the file name, so you can tell the snakes apart:
    saves/checkpoint.npz -> saves/checkpoint_vision.npz or saves/checkpoint_simple.npz"""
    name, ext = os.path.splitext(path)
    suffix = '_' + cfg.input_type
    if name.endswith(suffix):              # already named, e.g. when saving over a loaded file
        return path
    return name + suffix + (ext or '.npz')


def save(path, ga):
    """Save to `path` with the input type added to the name. Returns the file name used."""
    path = with_input_type(path, ga.cfg)
    best = ga.best_ever
    np.savez(path,
             config=json.dumps(ga.cfg.to_dict()),
             generation=ga.generation,
             population=np.array([ind.genome for ind in ga.population]),
             ages=np.array([ind.age for ind in ga.population]),
             history=json.dumps(ga.history),
             best_genome=best.genome if best else np.zeros(0),
             best_score=best.score if best else 0,
             best_generation=ga.best_ever_generation)
    return path


def load(path, settings=None):
    """Load the snakes. If `settings` is given, it is used for everything except the brain
    settings, which always come from the file (the saved genomes only fit that brain)."""
    data = np.load(path)
    cfg = Config.from_dict(json.loads(str(data['config'])))
    if settings is not None:
        cfg = settings.with_brain_of(cfg)
    ga = GeneticAlgorithm(cfg)
    ga.generation = int(data['generation'])
    ga.population = []
    for genome, age in zip(data['population'], data['ages']):
        ind = Individual(np.array(genome), born_by='loaded')
        ind.age = int(age)
        ga.population.append(ind)
    ga.history = json.loads(str(data['history']))
    if len(data['best_genome']):
        ga.best_ever = Individual(np.array(data['best_genome']), born_by='loaded')
        ga.best_ever.score = int(data['best_score'])
        ga.best_ever_generation = int(data['best_generation'])
    return ga
