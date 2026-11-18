"""The genetic algorithm: keep the best snakes, breed children from them, repeat.

One generation:
  1. every snake plays one game and gets a fitness  (done by the dashboard or headless.py)
  2. record_results()   -- statistics, best snake so far
  3. select_survivors() -- keep the fittest
  4. pick_parent() x2   -- choose a mother and a father among the survivors
  5. crossover()        -- mix their genomes into two children
  6. mutate()           -- change a few numbers in each child at random
  7. the survivors and the children form the next population
"""
import random
import numpy as np

from .network import random_genome, genome_size


class Individual:
    """One snake of the population: its genome and how well it did."""

    def __init__(self, genome, born_by='random'):
        self.genome = genome
        self.born_by = born_by    # 'random', 'SBX', 'single-point', 'two-point', 'uniform', 'clone', 'loaded'
        self.age = 0              # how many generations it has survived
        self.fitness = 0.0
        self.score = 0            # apples eaten
        self.steps = 0


# ---- crossover methods: two parent genomes -> two child genomes -----------
def sbx_crossover(mom, dad, eta):
    """Simulated binary crossover: each child number is a random blend of the two parents'
    numbers. A big eta keeps the children close to their parents."""
    u = np.random.random(len(mom))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    child1 = 0.5 * ((1 + beta) * mom + (1 - beta) * dad)
    child2 = 0.5 * ((1 - beta) * mom + (1 + beta) * dad)
    return child1, child2


def single_point_crossover(mom, dad, cut):
    """Child 1 = mom's genes before the cut + dad's genes after it. Child 2 the other way round."""
    child1 = np.concatenate([mom[:cut], dad[cut:]])
    child2 = np.concatenate([dad[:cut], mom[cut:]])
    return child1, child2


def two_point_crossover(mom, dad, cut1, cut2):
    """Swap the middle part between the two cuts."""
    child1 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
    child2 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
    return child1, child2


def uniform_crossover(mom, dad):
    """For every gene, flip a coin to decide which parent it comes from."""
    from_mom = np.random.random(len(mom)) < 0.5
    return np.where(from_mom, mom, dad), np.where(from_mom, dad, mom)


# ---- mutation methods: change some genes in place -------------------------
def gaussian_mutation(genome, rate, sigma):
    """Each gene, with probability `rate`, gets a small random nudge."""
    chosen = np.random.random(len(genome)) < rate
    genome[chosen] += np.random.normal(0, sigma, chosen.sum())


def uniform_mutation(genome, rate, limit):
    """Each gene, with probability `rate`, is replaced by a new random number in [-limit, limit]."""
    chosen = np.random.random(len(genome)) < rate
    genome[chosen] = np.random.uniform(-limit, limit, chosen.sum())


class GeneticAlgorithm:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.seed:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        self.generation = 0
        self.population = [Individual(random_genome(cfg.layer_sizes)) for _ in range(cfg.population_size)]
        self.history = []                 # one entry of statistics per finished generation
        self.best_ever = None             # the Individual with the most apples so far
        self.best_ever_generation = 0

    @property
    def genes(self):
        return genome_size(self.cfg.layer_sizes)

    @property
    def mutation_rate(self):
        if self.cfg.mutation_rate_type == 'decaying':
            return self.cfg.mutation_rate / np.sqrt(self.generation + 1)
        return self.cfg.mutation_rate

    # ---- step 2: statistics ------------------------------------------------
    def record_results(self):
        scores = [ind.score for ind in self.population]
        fitnesses = [ind.fitness for ind in self.population]
        self.history.append({'generation': self.generation,
                             'best_score': max(scores),
                             'avg_score': sum(scores) / len(scores),
                             'best_fitness': max(fitnesses)})
        champion = max(self.population, key=lambda ind: (ind.score, ind.fitness))
        if self.best_ever is None or champion.score > self.best_ever.score:
            self.best_ever = Individual(champion.genome.copy(), champion.born_by)
            self.best_ever.score = champion.score
            self.best_ever.fitness = champion.fitness
            self.best_ever_generation = self.generation

    # ---- step 3: survivors -------------------------------------------------
    def select_survivors(self):
        """The fittest `num_parents` snakes survive. Snakes older than `lifespan` may not."""
        candidates = self.population
        if self.cfg.lifespan > 0:
            young_enough = [ind for ind in candidates if ind.age < self.cfg.lifespan]
            candidates = young_enough or candidates
        candidates = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
        return candidates[:self.cfg.num_parents]

    # ---- step 4: choose a parent -------------------------------------------
    def pick_parent(self, survivors):
        if self.cfg.parent_selection == 'tournament':
            # pick a few at random, the fittest of them wins
            group = random.sample(survivors, min(self.cfg.tournament_size, len(survivors)))
            return max(group, key=lambda ind: ind.fitness)
        # roulette wheel: the chance of being picked is proportional to fitness
        lowest = min(ind.fitness for ind in survivors)
        weights = [ind.fitness - lowest + 1e-6 for ind in survivors]
        return random.choices(survivors, weights=weights)[0]

    # ---- step 5: crossover -------------------------------------------------
    def cut_position(self, fraction):
        """Where to cut the genome: at a fixed fraction of its length, or anywhere at random."""
        if self.cfg.cross_point_mode == 'fixed':
            return int(fraction * self.genes)
        return random.randint(1, self.genes - 1)

    def crossover(self, mom, dad):
        """Returns two child genomes and the name of the method used."""
        cfg = self.cfg
        if random.random() >= cfg.crossover_rate:
            return mom.genome.copy(), dad.genome.copy(), 'clone'
        if random.random() < cfg.sbx_ratio:
            child1, child2 = sbx_crossover(mom.genome, dad.genome, cfg.sbx_eta)
            return child1, child2, 'SBX'
        if cfg.point_crossover == 'uniform':
            child1, child2 = uniform_crossover(mom.genome, dad.genome)
            return child1, child2, 'uniform'
        if cfg.point_crossover == 'two':
            cut1, cut2 = sorted([self.cut_position(cfg.cross_point), self.cut_position(cfg.cross_point2)])
            child1, child2 = two_point_crossover(mom.genome, dad.genome, cut1, cut2)
            return child1, child2, 'two-point'
        cut = self.cut_position(cfg.cross_point)
        child1, child2 = single_point_crossover(mom.genome, dad.genome, cut)
        return child1, child2, 'single-point'

    # ---- step 6: mutation --------------------------------------------------
    def mutate(self, genome):
        if random.random() < self.cfg.gaussian_ratio:
            gaussian_mutation(genome, self.mutation_rate, self.cfg.gaussian_sigma)
        else:
            uniform_mutation(genome, self.mutation_rate, self.cfg.uniform_range)

    # ---- one whole generation step -----------------------------------------
    def next_generation(self):
        """Call this after every snake of the population has played and has a fitness."""
        self.record_results()
        survivors = self.select_survivors()

        children = []
        while len(children) < self.cfg.num_offspring:
            mom = self.pick_parent(survivors)
            dad = self.pick_parent(survivors)
            genome1, genome2, method = self.crossover(mom, dad)
            for genome in (genome1, genome2):
                self.mutate(genome)
                children.append(Individual(genome, born_by=method))
        children = children[:self.cfg.num_offspring]

        if self.cfg.selection_type == 'plus':
            # (mu + lambda): survivors stay and compete with their children next generation
            for ind in survivors:
                ind.age += 1
            self.population = survivors + children
        else:
            # (mu, lambda): only the children make up the next generation
            self.population = children
        self.generation += 1
