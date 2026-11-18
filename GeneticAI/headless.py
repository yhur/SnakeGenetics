"""Fast training without any window: every snake of a generation plays in parallel.

This is the only parallel code in the project. The genetic algorithm is the same one
the dashboard uses; the only difference is how the games are played:

    dashboard: one snake after another, on screen
    headless : a pool of worker processes (one per CPU core) plays all of them at once
"""
import multiprocessing
import os
import signal
import time

from . import checkpoint
from .snake_ai import play_game


def ignore_ctrl_c():
    """Workers ignore Ctrl+C; the main process handles it and stops cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train(ga, generations=0, workers=None, save_path='saves/checkpoint.npz', save_every=10):
    """Train until `generations` more generations are done (0 = until Ctrl+C)."""
    workers = workers or os.cpu_count()
    save_path = checkpoint.with_input_type(save_path, ga.cfg)   # e.g. saves/checkpoint_simple.npz
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    print(f'Training {ga.cfg.population_size} snakes per generation on {workers} processes. '
          f'Ctrl+C to stop. Saving to {save_path}')
    print(f'Network {ga.cfg.layer_sizes}, {ga.cfg.parent_selection} selection, '
          f'{ga.cfg.sbx_ratio:.0%} SBX / {1 - ga.cfg.sbx_ratio:.0%} '
          f'{ {"single": "single-point", "two": "two-point", "uniform": "uniform"}[ga.cfg.point_crossover]}, '
          f'mutation {ga.cfg.mutation_rate}')

    pool = multiprocessing.Pool(workers, initializer=ignore_ctrl_c)
    done = 0
    try:
        while generations == 0 or done < generations:
            start = time.time()

            # Play every snake's game, in parallel. results[i] belongs to population[i].
            jobs = [(ga.cfg, ind.genome) for ind in ga.population]
            results = pool.starmap(play_game, jobs)
            for ind, (fitness, score, steps) in zip(ga.population, results):
                ind.fitness, ind.score, ind.steps = fitness, score, steps

            ga.next_generation()          # same genetic algorithm as the dashboard
            done += 1

            h = ga.history[-1]
            print(f'gen {h["generation"]:5d} | best {h["best_score"]:3d} | avg {h["avg_score"]:6.2f} | '
                  f'best ever {ga.best_ever.score:3d} (gen {ga.best_ever_generation}) | '
                  f'{time.time() - start:5.2f} s', flush=True)
            if done % save_every == 0:
                checkpoint.save(save_path, ga)
    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        pool.terminate()
        checkpoint.save(save_path, ga)
        print(f'Saved generation {ga.generation} to {save_path}')
