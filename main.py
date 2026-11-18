#!/usr/bin/env python
"""Teach a neural network to play SnakeGame with a genetic algorithm.

    python main.py                         dashboard: watch it learn, one snake at a time
    python main.py --headless              no window: train as fast as possible (parallel)
    python main.py --headless -g 500       ... stop after 500 generations
    python main.py --load saves/x.npz      continue from a checkpoint (either mode)

Both modes use settings.json: the settings last applied in the dashboard. With --load, the
brain settings (layers, neurons, activations, inputs, vision) come from the checkpoint instead.

Dashboard keys:  Space = pause / one move    R = resume    Q = quit
"""
import argparse
import sys

from GeneticAI.config import Config
from GeneticAI.genetic import GeneticAlgorithm
from GeneticAI import checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--headless', action='store_true', help='train without a window, in parallel')
    parser.add_argument('-g', '--generations', type=int, default=0, help='headless: generations to run (0 = until Ctrl+C)')
    parser.add_argument('--workers', type=int, help='headless: processes to use (default: all CPU cores)')
    parser.add_argument('--save', default='saves/checkpoint.npz', help='headless: checkpoint file')
    parser.add_argument('--config', help='JSON settings file (default: settings.json, written by the dashboard)')
    parser.add_argument('--load', help='checkpoint to continue from')
    args = parser.parse_args()

    settings = Config.load(args.config) if args.config else Config.load_last()
    if args.load:
        ga = checkpoint.load(args.load, settings)     # snakes + brain from the file, the rest from settings
    else:
        ga = GeneticAlgorithm(settings)

    if args.headless:
        from GeneticAI import headless
        headless.train(ga, args.generations, args.workers, args.save)
    else:
        from PyQt5.QtWidgets import QApplication
        from Dashboard.main_window import Dashboard
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = Dashboard(ga)
        window.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
