"""The dashboard: watch the genetic algorithm train, one snake at a time.

The whole training loop is in this file:

    for each generation:
        for each snake in the population:      <- start_snake() / snake_finished()
            play its game on screen, move by move
            its fitness = how well it did
        ga.next_generation()                    <- selection, crossover, mutation

Keys:  Space = pause, or one move while paused    R = resume    Q = quit
"""
import os
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel,
                             QPushButton, QComboBox, QSplitter, QFileDialog, QMessageBox, QApplication,
                             QCheckBox, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QEvent

from GeneticAI.genetic import GeneticAlgorithm
from GeneticAI.config import SETTINGS_FILE
from GeneticAI.snake_ai import SnakeAI, input_labels, MOVE_NAMES
from GeneticAI import checkpoint
from .nn_view import NetworkView
from .board_view import BoardView
from .chart_view import ChartView
from .config_panel import ConfigPanel

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saves')
SPEEDS = [('1 move/s', 1), ('5 moves/s', 5), ('15 moves/s', 15), ('30 moves/s', 30), ('60 moves/s', 60),
          ('200 moves/s', 200), ('1000 moves/s', 1000), ('Max', 0)]


class Dashboard(QMainWindow):
    def __init__(self, ga):
        super().__init__()
        self.setWindowTitle('SnakeGenetics — Genetic Algorithm')
        self.ga = ga
        self.index = 0            # which snake of the population is playing now
        self.game = None          # its game
        self.paused = False
        self.move_credit = 0.0    # moves owed to the speed setting (see on_timer)
        self.last_time = time.time()
        self.previous = '-'       # how the previous snake did

        self.build_window()

        QApplication.instance().installEventFilter(self)   # catch Space / R / Q everywhere
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)                               # about 60 times a second
        self.start_snake()

    # =====================================================================
    #   The training loop
    # =====================================================================
    def start_snake(self):
        """Start the game of snake number `index` of the current generation."""
        snake = self.ga.population[self.index]
        self.game = SnakeAI(self.ga.cfg, snake.genome)
        self.board.show_game(self.game)
        self.nn_view.show_brain(self.game.brain, input_labels(self.ga.cfg), MOVE_NAMES)

    def one_move(self):
        self.game.step()
        if self.game.done:
            self.snake_finished()

    def snake_finished(self):
        """The snake died: store how well it did, then go on to the next snake."""
        snake = self.ga.population[self.index]
        snake.fitness = self.game.fitness()
        snake.score = self.game.score
        snake.steps = self.game.steps
        self.previous = f'{snake.score} apples, {self.game.status}'

        self.index += 1
        if self.index == len(self.ga.population):    # every snake has played:
            self.ga.next_generation()                # selection, crossover, mutation
            self.index = 0
            self.chart.set_history(self.ga.history)
        self.start_snake()

    def on_timer(self):
        """Called about 60 times a second. Makes as many moves as the chosen speed asks for."""
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        if self.paused:
            return
        speed = self.speed.currentData()
        if speed == 0:
            # Max: keep moving for 12 milliseconds, then let the screen redraw
            while time.time() - now < 0.012:
                self.one_move()
        else:
            self.move_credit += speed * elapsed
            while self.move_credit >= 1:
                self.move_credit -= 1
                self.one_move()
        self.refresh()

    # =====================================================================
    #   Keys
    # =====================================================================
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and self.isActiveWindow():
            if event.key() == Qt.Key_Space:
                self.pause_or_step()
                return True
            if event.key() == Qt.Key_R:
                self.resume()
                return True
            if event.key() == Qt.Key_Q:
                self.close()
                return True
        return super().eventFilter(obj, event)

    def pause_or_step(self):
        if self.paused:
            self.one_move()          # already paused: make one move
        self.paused = True
        self.refresh()

    def resume(self):
        self.paused = False
        self.move_credit = 0
        self.refresh()

    # =====================================================================
    #   Screen
    # =====================================================================
    def refresh(self):
        ga, game, s = self.ga, self.game, self.stat
        s['generation'].setText(str(ga.generation))
        s['snake'].setText(f'{self.index + 1} / {len(ga.population)}')
        if ga.best_ever:
            s['best'].setText(f'{ga.best_ever.score}  (generation {ga.best_ever_generation})')
        if ga.history:
            last = ga.history[-1]
            s['last_best'].setText(str(last['best_score']))
            s['last_avg'].setText(f'{last["avg_score"]:.2f}')
            s['last_fitness'].setText(f'{last["best_fitness"]:.0f}')
        s['mutation'].setText(f'{ga.mutation_rate:.4f}')
        s['genes'].setText(str(ga.genes))

        snake = ga.population[self.index]
        s['score'].setText(str(game.score))
        s['steps'].setText(str(game.steps))
        s['born'].setText(snake.born_by)
        s['age'].setText(f'{snake.age}' + ('  (survived from earlier)' if snake.age else '  (new)'))
        s['next'].setText(MOVE_NAMES[int(game.outputs.argmax())])
        s['outputs'].setText(' / '.join(f'{v:.2f}' for v in game.outputs))
        s['previous'].setText(self.previous)

        if self.paused:
            self.state.setText('PAUSED — Space: one move, R: resume')
            self.state.setStyleSheet('font-weight: bold; color: #e6a800;')
        else:
            self.state.setText('RUNNING — Space: pause, Q: quit')
            self.state.setStyleSheet('font-weight: bold; color: #2bb24c;')
        self.board.update()
        self.nn_view.update()

    def build_window(self):
        cfg = self.ga.cfg
        self.nn_view = NetworkView()
        self.board = BoardView(cfg)
        self.chart = ChartView()
        self.chart.set_history(self.ga.history)
        self.panel = ConfigPanel(cfg)
        self.panel.applied.connect(self.restart)

        # tool bar
        pause = QPushButton('Pause / Step  [Space]')
        resume = QPushButton('Resume  [R]')
        quit_ = QPushButton('Quit  [Q]')
        save = QPushButton('Save')
        load = QPushButton('Load...')
        pause.clicked.connect(self.pause_or_step)
        resume.clicked.connect(self.resume)
        quit_.clicked.connect(self.close)
        save.clicked.connect(self.save)
        load.clicked.connect(self.load_dialog)
        self.speed = QComboBox()
        for label, value in SPEEDS:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(3)
        vision = QCheckBox('Vision rays')
        vision.setChecked(True)
        vision.toggled.connect(self.toggle_vision)
        self.state = QLabel()
        bar = QHBoxLayout()
        for w in (pause, resume, quit_, self.speed, vision, save, load):
            w.setFocusPolicy(Qt.NoFocus)
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.state)

        # statistics
        self.stat = {}
        left = [('generation', 'Generation'), ('snake', 'Snake playing'), ('best', 'Best score ever'),
                ('last_best', 'Best score, last gen'), ('last_avg', 'Average score, last gen'),
                ('last_fitness', 'Best fitness, last gen'), ('mutation', 'Mutation rate'),
                ('genes', 'Genes per snake')]
        right = [('score', 'Score'), ('steps', 'Steps'), ('born', 'Born by'), ('age', 'Age (generations)'),
                 ('next', 'Next move'), ('outputs', 'Outputs F / R / L'), ('previous', 'Previous snake')]
        stats = QGroupBox('Training')
        grid = QGridLayout(stats)
        for column, rows in ((0, left), (2, right)):
            for row, (key, label) in enumerate(rows):
                grid.addWidget(QLabel(f'<b>{label}:</b>'), row, column)
                value = QLabel('-')
                value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # text changes don't resize panes
                grid.addWidget(value, row, column + 1)
                self.stat[key] = value
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        center = QWidget()
        column = QVBoxLayout(center)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.board, 5)
        column.addWidget(stats, 0)
        column.addWidget(self.chart, 2)

        panes = QSplitter(Qt.Horizontal)
        panes.addWidget(self.nn_view)
        panes.addWidget(center)
        panes.addWidget(self.panel)
        panes.setSizes([700, 560, 360])
        panes.setChildrenCollapsible(False)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(bar)
        layout.addWidget(panes, 1)
        self.setCentralWidget(root)
        self.resize(1680, 980)

    def toggle_vision(self, on):
        self.board.show_vision = on
        self.board.update()

    # =====================================================================
    #   Restart, save, load
    # =====================================================================
    def restart(self, cfg):
        """Apply the panel's settings. If the brain shape is unchanged, the current snakes can be kept."""
        box = QMessageBox(self)
        box.setWindowTitle('Apply settings')
        if cfg.same_brain_as(self.ga.cfg):
            box.setText('Keep the current snakes and use the new settings from now on,\n'
                        'or start a new random population?')
            keep = box.addButton('Keep snakes', QMessageBox.AcceptRole)
        else:
            box.setText('The brain shape changed (layers, neurons, activations or inputs),\n'
                        'so the current snakes cannot be kept. Start a new random population?')
            keep = None
        new = box.addButton('New population', QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()

        if keep is not None and box.clickedButton() == keep:
            self.ga.cfg = cfg                      # same snakes, new rules from the next step on
        elif box.clickedButton() == new:
            self.use(GeneticAlgorithm(cfg))
        else:
            return
        cfg.save(SETTINGS_FILE)                    # headless mode uses these settings too
        self.statusBar().showMessage(f'Settings saved to {SETTINGS_FILE}', 6000)
        self.refresh()

    def use(self, ga):
        self.ga = ga
        self.index = 0
        self.previous = '-'
        self.chart.set_history(ga.history)
        self.start_snake()
        self.refresh()

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f'checkpoint_gen{self.ga.generation}.npz')
        path = checkpoint.save(path, self.ga)          # adds _vision or _simple to the name
        self.statusBar().showMessage(f'Saved {path}', 6000)

    def load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load checkpoint', SAVE_DIR, 'Checkpoint (*.npz)')
        if path:
            ga = checkpoint.load(path, self.ga.cfg)   # brain from the file, everything else as now
            self.panel.load_config(ga.cfg)
            ga.cfg.save(SETTINGS_FILE)
            self.use(ga)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
