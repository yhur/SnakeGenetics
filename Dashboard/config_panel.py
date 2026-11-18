"""Editable settings. 'Apply' uses them, keeping the current snakes when the brain shape is unchanged."""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QPushButton, QLabel, QScrollArea)
from PyQt5.QtCore import pyqtSignal, Qt

from GeneticAI.config import Config
from GeneticAI.network import genome_size


def _spin(lo, hi, val, step=1):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _dspin(lo, hi, val, step, decimals=3):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _combo(items, val):
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    c.setCurrentIndex(max(0, c.findData(val)))
    return c


class ConfigPanel(QScrollArea):
    applied = pyqtSignal(object)          # emits a Config

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(330)
        body = QWidget()
        self.setWidget(body)
        v = QVBoxLayout(body)

        # ---------------- Network ----------------
        nn = QGroupBox('Neural Network')
        fn = QFormLayout(nn)
        self.n_layers = _spin(0, 6, len(cfg.hidden_layers))
        self.n_layers.valueChanged.connect(self._rebuild_layer_spins)
        fn.addRow('Hidden layers', self.n_layers)
        self.layer_box = QWidget()
        self.layer_form = QFormLayout(self.layer_box)
        self.layer_form.setContentsMargins(0, 0, 0, 0)
        self.layer_spins = []
        fn.addRow(self.layer_box)
        self.hidden_act = _combo([('ReLU', 'relu'), ('Leaky ReLU', 'leaky_relu'), ('Tanh', 'tanh'),
                                  ('Sigmoid', 'sigmoid')], cfg.hidden_activation)
        self.output_act = _combo([('Sigmoid', 'sigmoid'), ('Softmax', 'softmax'), ('Linear', 'linear')],
                                 cfg.output_activation)
        self.input_type = _combo([('Vision', 'vision'), ('Simple (14 facts)', 'simple')], cfg.input_type)
        self.vision = _combo([('8 directions', 8), ('4 directions', 4)], cfg.vision_directions)
        self.as_vision = _combo([('Binary', 'binary'), ('1 / distance', 'distance')], cfg.apple_self_vision)
        fn.addRow('Hidden activation', self.hidden_act)
        fn.addRow('Output activation', self.output_act)
        fn.addRow('Inputs', self.input_type)
        fn.addRow('Snake vision', self.vision)
        fn.addRow('Apple/Self vision', self.as_vision)
        self.arch_label = QLabel()
        fn.addRow('Architecture', self.arch_label)
        v.addWidget(nn)

        # ---------------- Selection ----------------
        sel = QGroupBox('Population && Selection')
        fs = QFormLayout(sel)
        self.parents = _spin(2, 5000, cfg.num_parents, 10)
        self.offspring = _spin(2, 20000, cfg.num_offspring, 10)
        self.sel_type = _combo([('Plus (μ+λ)', 'plus'), ('Comma (μ,λ)', 'comma')], cfg.selection_type)
        self.parent_sel = _combo([('Roulette wheel', 'roulette'), ('Tournament', 'tournament')],
                                 cfg.parent_selection)
        self.tourn = _spin(2, 100, cfg.tournament_size)
        self.lifespan = _spin(0, 10000, cfg.lifespan)
        self.lifespan.setSpecialValueText('infinite')
        fs.addRow('Parents (μ)', self.parents)
        fs.addRow('Offspring (λ)', self.offspring)
        fs.addRow('Selection type', self.sel_type)
        fs.addRow('Parent selection', self.parent_sel)
        fs.addRow('Tournament size', self.tourn)
        fs.addRow('Lifespan (gens)', self.lifespan)
        v.addWidget(sel)

        # ---------------- Crossover ----------------
        cx = QGroupBox('Crossover')
        fc = QFormLayout(cx)
        self.cx_rate = _dspin(0, 1, cfg.crossover_rate, 0.05, 2)
        self.sbx_ratio = _dspin(0, 1, cfg.sbx_ratio, 0.05, 2)
        self.sbx_eta = _dspin(0.1, 1000, cfg.sbx_eta, 5, 1)
        self.point_type = _combo([('Single-point', 'single'), ('Two-point', 'two'), ('Uniform', 'uniform')],
                                 cfg.point_crossover)
        self.cut_mode = _combo([('Random', 'random'), ('Fixed', 'fixed')], cfg.cross_point_mode)
        self.cut1 = _dspin(0, 1, cfg.cross_point, 0.05, 2)
        self.cut2 = _dspin(0, 1, cfg.cross_point2, 0.05, 2)
        fc.addRow('Crossover rate', self.cx_rate)
        fc.addRow('SBX share', self.sbx_ratio)
        fc.addRow('SBX η', self.sbx_eta)
        fc.addRow('Point crossover', self.point_type)
        fc.addRow('Cross point', self.cut_mode)
        fc.addRow('Cross point 1 (0-1)', self.cut1)
        fc.addRow('Cross point 2 (0-1)', self.cut2)
        self.cx_note = QLabel()
        self.cx_note.setWordWrap(True)
        self.cx_note.setStyleSheet('color: #999;')
        fc.addRow(self.cx_note)
        v.addWidget(cx)

        # ---------------- Mutation ----------------
        mu = QGroupBox('Mutation')
        fm = QFormLayout(mu)
        self.mut_rate = _dspin(0, 1, cfg.mutation_rate, 0.005, 3)
        self.mut_type = _combo([('Static', 'static'), ('Decaying (rate/√gen)', 'decaying')],
                               cfg.mutation_rate_type)
        self.gauss_ratio = _dspin(0, 1, cfg.gaussian_ratio, 0.05, 2)
        self.sigma = _dspin(0.001, 5, cfg.gaussian_sigma, 0.05, 3)
        self.urange = _dspin(0.01, 10, cfg.uniform_range, 0.1, 2)
        fm.addRow('Mutation rate', self.mut_rate)
        fm.addRow('Rate type', self.mut_type)
        fm.addRow('Gaussian share', self.gauss_ratio)
        fm.addRow('Gaussian σ', self.sigma)
        fm.addRow('Uniform range ±', self.urange)
        v.addWidget(mu)

        # ---------------- Game ----------------
        gm = QGroupBox('Game')
        fg = QFormLayout(gm)
        self.gx = _spin(5, 80, cfg.grid_x)
        self.gy = _spin(5, 80, cfg.grid_y)
        self.seed = _spin(0, 2 ** 31 - 1, cfg.seed)
        self.seed.setSpecialValueText('random')
        fg.addRow('Grid width', self.gx)
        fg.addRow('Grid height', self.gy)
        fg.addRow('Random seed', self.seed)
        v.addWidget(gm)

        row = QHBoxLayout()
        self.apply_btn = QPushButton('Apply')
        self.apply_btn.setStyleSheet('font-weight: bold; padding: 6px;')
        self.apply_btn.clicked.connect(lambda: self.applied.emit(self.config()))
        reset = QPushButton('Defaults')
        reset.clicked.connect(lambda: self.load_config(Config()))
        row.addWidget(reset)
        row.addWidget(self.apply_btn)
        v.addLayout(row)
        v.addStretch(1)

        for w in self._all_inputs():
            sig = getattr(w, 'valueChanged', None) or w.currentIndexChanged
            sig.connect(self._refresh_labels)
        self._rebuild_layer_spins(len(cfg.hidden_layers), cfg.hidden_layers)

    # ------------------------------------------------------------------
    def _all_inputs(self):
        return [self.hidden_act, self.output_act, self.input_type, self.vision, self.as_vision, self.parents, self.offspring,
                self.sel_type, self.parent_sel, self.cut_mode, self.point_type, self.cut1, self.cut2,
                self.sbx_ratio]

    def _rebuild_layer_spins(self, n, values=None):
        old = [s.value() for s in self.layer_spins]
        values = list(values) if values is not None else old
        while self.layer_form.rowCount():
            self.layer_form.removeRow(0)
        self.layer_spins = []
        for i in range(n):
            val = values[i] if i < len(values) else (values[-1] if values else 16)
            s = _spin(1, 256, val)
            s.setFocusPolicy(Qt.ClickFocus)
            s.valueChanged.connect(self._refresh_labels)
            self.layer_form.addRow(f'  Neurons in layer {i + 1}', s)
            self.layer_spins.append(s)
        self._refresh_labels()

    def _refresh_labels(self, *_):
        cfg = self.config()
        n = genome_size(cfg.layer_sizes)
        self.arch_label.setText(f'{cfg.layer_sizes}  ({n} genes)')
        vision = cfg.input_type == 'vision'
        self.vision.setEnabled(vision)
        self.as_vision.setEnabled(vision)
        self.tourn.setEnabled(cfg.parent_selection == 'tournament')
        self.lifespan.setEnabled(cfg.selection_type == 'plus')      # with comma, no snake survives to age
        point = cfg.sbx_ratio < 1
        self.point_type.setEnabled(point)
        fixed = point and cfg.point_crossover != 'uniform'
        self.cut_mode.setEnabled(fixed)
        self.cut1.setEnabled(fixed and cfg.cross_point_mode == 'fixed')
        self.cut2.setEnabled(fixed and cfg.cross_point_mode == 'fixed' and cfg.point_crossover == 'two')
        self.cx_note.setText(
            f'{cfg.sbx_ratio:.0%} SBX / {1 - cfg.sbx_ratio:.0%} {cfg.point_crossover}-point. '
            'A fixed cross point of 0.5 cuts the genome in the middle: child 1 gets '
            'the first half from parent 1 and the second half from parent 2. '
            f'Population per generation: {cfg.population_size}.')

    def config(self) -> Config:
        return Config(
            grid_x=self.gx.value(), grid_y=self.gy.value(),
            hidden_layers=[s.value() for s in self.layer_spins],
            hidden_activation=self.hidden_act.currentData(), output_activation=self.output_act.currentData(),
            input_type=self.input_type.currentData(),
            vision_directions=self.vision.currentData(), apple_self_vision=self.as_vision.currentData(),
            num_parents=self.parents.value(), num_offspring=self.offspring.value(),
            selection_type=self.sel_type.currentData(), parent_selection=self.parent_sel.currentData(),
            tournament_size=self.tourn.value(), crossover_rate=self.cx_rate.value(),
            sbx_ratio=self.sbx_ratio.value(), sbx_eta=self.sbx_eta.value(),
            point_crossover=self.point_type.currentData(), cross_point_mode=self.cut_mode.currentData(),
            cross_point=self.cut1.value(), cross_point2=self.cut2.value(),
            mutation_rate=self.mut_rate.value(), mutation_rate_type=self.mut_type.currentData(),
            gaussian_ratio=self.gauss_ratio.value(), gaussian_sigma=self.sigma.value(),
            uniform_range=self.urange.value(), lifespan=self.lifespan.value(), seed=self.seed.value())

    def load_config(self, cfg: Config):
        for w in self.findChildren(QWidget):
            w.blockSignals(True)
        self.n_layers.setValue(len(cfg.hidden_layers))
        for combo, val in [(self.hidden_act, cfg.hidden_activation), (self.output_act, cfg.output_activation),
                           (self.input_type, cfg.input_type), (self.vision, cfg.vision_directions), (self.as_vision, cfg.apple_self_vision),
                           (self.sel_type, cfg.selection_type), (self.parent_sel, cfg.parent_selection),
                           (self.point_type, cfg.point_crossover), (self.cut_mode, cfg.cross_point_mode),
                           (self.mut_type, cfg.mutation_rate_type)]:
            combo.setCurrentIndex(max(0, combo.findData(val)))
        for spin, val in [(self.parents, cfg.num_parents), (self.offspring, cfg.num_offspring),
                          (self.tourn, cfg.tournament_size), (self.lifespan, cfg.lifespan),
                          (self.cx_rate, cfg.crossover_rate), (self.sbx_ratio, cfg.sbx_ratio),
                          (self.sbx_eta, cfg.sbx_eta), (self.cut1, cfg.cross_point), (self.cut2, cfg.cross_point2),
                          (self.mut_rate, cfg.mutation_rate), (self.gauss_ratio, cfg.gaussian_ratio),
                          (self.sigma, cfg.gaussian_sigma), (self.urange, cfg.uniform_range),
                          (self.gx, cfg.grid_x), (self.gy, cfg.grid_y), (self.seed, cfg.seed)]:
            spin.setValue(val)
        for w in self.findChildren(QWidget):
            w.blockSignals(False)
        self._rebuild_layer_spins(len(cfg.hidden_layers), cfg.hidden_layers)
