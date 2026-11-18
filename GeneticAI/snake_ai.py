"""One snake playing one game: it looks around, its brain decides, it moves.

Two kinds of inputs (setting 'input_type'):
  vision   the snake looks along 8 (or 4) rays and sees the wall, the apple and its own body;
           plus the head and the tail direction. 32 numbers.
  simple   14 yes/no facts: a wall forward/right/left?, the moving direction,
           the snake's own body forward/right/left?, which side the apple is.

The game itself is the original, unmodified code in ../SnakeGame (snake.Snake and boards.Board).
"""
import os
import sys
import numpy as np

# Make the original game importable.
GAME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SnakeGame')
sys.path.insert(0, GAME_DIR)
from snake import Snake, Direction, Point   # noqa: E402
from boards import Board             # noqa: E402

from .network import NeuralNetwork   # noqa: E402

# The game's own move encoding: [forward, right, left]. Output neuron i means MOVES[i].
MOVES = [Snake.action_f, Snake.action_r, Snake.action_l]
MOVE_NAMES = ['Forward', 'Right', 'Left']

# Directions the snake looks in, as (dx, dy). y grows downward on screen.
RAYS_8 = {'U': (0, -1), 'UR': (1, -1), 'R': (1, 0), 'DR': (1, 1),
          'D': (0, 1), 'DL': (-1, 1), 'L': (-1, 0), 'UL': (-1, -1)}
RAYS_4 = {'U': (0, -1), 'R': (1, 0), 'D': (0, 1), 'L': (-1, 0)}
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

# The game lists the directions clockwise: RIGHT, DOWN, LEFT, UP.
# So "turn right" is the next one in this list, and "turn left" the previous one.
CLOCKWISE = list(Direction)

SIMPLE_LABELS = ['Wall F', 'Wall R', 'Wall L',
                 'Moving R', 'Moving D', 'Moving L', 'Moving U',
                 'Body F', 'Body R', 'Body L',
                 'Apple L', 'Apple R', 'Apple U', 'Apple D']


def input_labels(cfg):
    """Names of the input neurons. Vision: 'U-W' = wall looking up, 'U-A' = apple, 'U-S' = self."""
    if cfg.input_type == 'simple':
        return SIMPLE_LABELS
    rays = RAYS_8 if cfg.vision_directions == 8 else RAYS_4
    labels = []
    for name in rays:
        labels += [name + '-W', name + '-A', name + '-S']
    return labels + ['HU', 'HD', 'HL', 'HR', 'TU', 'TD', 'TL', 'TR']


def fitness_function(steps, apples, wasted_steps):
    """How good was this game? Apples count a lot, survival a little.

    wasted_steps: moves spent circling before starving. They earn nothing,
    otherwise snakes learn to just circle until the game's starvation rule ends them.
    """
    useful_steps = steps - wasted_steps
    return useful_steps + (2 ** apples + 500 * apples ** 2.1) - (apples ** 1.2 * (0.25 * steps) ** 1.3)


class SnakeAI:
    def __init__(self, cfg, genome):
        self.cfg = cfg
        self.brain = NeuralNetwork(cfg.layer_sizes, genome, cfg.hidden_activation, cfg.output_activation)
        self.rays = RAYS_8 if cfg.vision_directions == 8 else RAYS_4

        # A fresh game, using the original Board and Snake.
        self.board = Board(cfg.grid_x, cfg.grid_y)
        self.snake = Snake(cfg.grid_x, cfg.grid_y)
        self.board.snake = self.snake
        self.board.generate_apple(self.snake)

        self.steps = 0
        self.status = 'ok'          # 'ok', 'apple', or how it ended: 'wall', 'body', 'starved', 'won'
        self.done = False
        self.move = None            # index of the last move made (0 forward, 1 right, 2 left)
        self.ray_hits = []          # what each ray saw, for drawing
        self.seen = set()           # every state since the last apple, to catch endless loops
        self.think()                # decide the first move

    @property
    def score(self):
        return self.snake.score

    # ---- 1. look ----------------------------------------------------------
    def look(self):
        """Build the input list for the brain."""
        if self.cfg.input_type == 'simple':
            return self.look_simple()
        return self.look_vision()

    def look_vision(self):
        """For every ray: 1/distance to the wall, whether the apple is on the ray, whether
        the snake's own body is on the ray. Then the head direction and the tail
        direction, each as 4 numbers where only one is 1 (up, down, left, right).
        """
        head_x, head_y = int(self.snake.head.x), int(self.snake.head.y)
        body = [(int(p.x), int(p.y)) for p in self.snake.body[1:]]
        apple = (int(self.board.apple.x), int(self.board.apple.y))
        binary = self.cfg.apple_self_vision == 'binary'

        inputs = []
        self.ray_hits = []
        for dx, dy in self.rays.values():
            # walk along the ray until we leave the board
            x, y, distance = head_x + dx, head_y + dy, 1
            apple_distance, body_distance = 0, 0
            while 0 <= x < self.cfg.grid_x and 0 <= y < self.cfg.grid_y:
                if apple_distance == 0 and (x, y) == apple:
                    apple_distance = distance
                if body_distance == 0 and (x, y) in body:
                    body_distance = distance
                x, y, distance = x + dx, y + dy, distance + 1
            self.ray_hits.append((dx, dy, distance, apple_distance, body_distance))

            inputs.append(1 / distance)                       # wall: the closer, the bigger
            if binary:
                inputs.append(1 if apple_distance else 0)
                inputs.append(1 if body_distance else 0)
            else:
                inputs.append(1 / apple_distance if apple_distance else 0)
                inputs.append(1 / body_distance if body_distance else 0)

        for d in DIRECTIONS:                                  # which way the head is going
            inputs.append(1 if self.snake.direction == d else 0)
        tail, before_tail = self.snake.body[-1], self.snake.body[-2]
        tail_direction = (before_tail.x - tail.x, before_tail.y - tail.y)
        for d in DIRECTIONS:                                  # which way the tail is going
            inputs.append(1 if tail_direction == d.value else 0)
        return inputs

    def cells_around_head(self):
        """The cells straight ahead, to the right and to the left of the head."""
        i = CLOCKWISE.index(self.snake.direction)
        cells = []
        for direction in (CLOCKWISE[i], CLOCKWISE[(i + 1) % 4], CLOCKWISE[(i - 1) % 4]):
            dx, dy = direction.value
            cells.append(Point(self.snake.head.x + dx, self.snake.head.y + dy))
        return cells                     # [forward, right, left]

    def is_wall(self, cell):
        """Is this cell outside the board?"""
        return not (0 <= cell.x < self.cfg.grid_x and 0 <= cell.y < self.cfg.grid_y)

    def is_body(self, cell):
        """Is the snake's own body in this cell?"""
        return cell in self.snake.body

    def is_danger(self, cell):
        """Would moving into this cell end the game? (a wall, or the snake's own body)"""
        return self.is_wall(cell) or self.is_body(cell)

    def look_simple(self):
        """14 yes/no facts (1 or 0) about the cells next to the head and where the apple is."""
        head, apple, direction = self.snake.head, self.board.apple, self.snake.direction
        cells = self.cells_around_head()                                      # forward, right, left
        inputs = [self.is_wall(cell) for cell in cells]                       # wall F, R, L
        inputs += [direction == Direction.RIGHT, direction == Direction.DOWN, # moving R, D, L, U
                   direction == Direction.LEFT, direction == Direction.UP]
        inputs += [self.is_body(cell) for cell in cells]                      # body F, R, L
        inputs += [apple.x < head.x, apple.x > head.x,                        # apple L, R, U, D
                   apple.y < head.y, apple.y > head.y]
        return [1 if fact else 0 for fact in inputs]

    # ---- 2. think ---------------------------------------------------------
    def think(self):
        self.inputs = self.look()
        self.outputs = self.brain.forward(self.inputs)

    # ---- 3. move ----------------------------------------------------------
    def step(self):
        """Make one move: the output neuron with the highest value wins."""
        if self.done:
            return
        self.move = int(np.argmax(self.outputs))
        self.status = self.snake.moveTo(MOVES[self.move])
        self.steps += 1

        if self.status == 'apple':
            if len(self.snake.body) == self.cfg.grid_x * self.cfg.grid_y:
                self.status, self.done = 'won', True          # no room left for an apple
                return
            self.board.generate_apple(self.snake)
            self.seen.clear()                                 # a longer snake can't repeat an older state
        elif self.status != 'ok':                             # 'wall', 'body' or 'starved'
            self.done = True
            return

        if self.is_looping():
            self.end_as_loop()
            return
        self.think()                                          # decide the next move

    # ---- endless-loop check -----------------------------------------------
    def is_looping(self):
        """True if the snake is exactly where it was before, since its last apple.

        The brain always makes the same move for the same situation, and nothing random happens
        until an apple is eaten. So once a state comes back, the snake will repeat the same moves
        forever and starve. The state is the whole body, in order, plus the direction.
        """
        state = (self.snake.direction, tuple(self.snake.body))
        if state in self.seen:
            return True
        self.seen.add(state)
        return False

    def end_as_loop(self):
        """End the game now, but score it exactly as if the snake had circled until it starved."""
        starves_at = 100 * (len(self.snake.body) + 1) + 201   # the game's own starvation rule
        remaining = starves_at - self.snake.moves
        self.steps += remaining
        self.snake.moves += remaining
        self.status, self.done = 'loop', True

    def fitness(self):
        wasted = self.snake.moves if self.status in ('starved', 'loop') else 0
        return fitness_function(self.steps, self.score, wasted)


def play_game(cfg, genome):
    """Play one whole game without drawing anything. Returns (fitness, apples, steps)."""
    game = SnakeAI(cfg, genome)
    while not game.done:
        game.step()
    return game.fitness(), game.score, game.steps
