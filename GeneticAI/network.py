"""The snake's brain: a small neural network whose weights come from a genome.

A genome is just one long list of numbers. The network cuts it into weight
matrices and bias vectors, one per layer.
"""
import numpy as np


# ---- activation functions -------------------------------------------------
def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def sigmoid(x):
    x = np.clip(x, -500, 500)       # e^500 still fits in a float; sigmoid is already 0 or 1 out there
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def linear(x):
    return x


ACTIVATIONS = {'relu': relu, 'leaky_relu': leaky_relu, 'sigmoid': sigmoid,
               'tanh': tanh, 'softmax': softmax, 'linear': linear}


# ---- genome helpers -------------------------------------------------------
def genome_size(layer_sizes):
    """How many numbers a genome needs: every weight plus every bias.

    Example: layers [32, 20, 12, 3] need (32*20 + 20) + (20*12 + 12) + (12*3 + 3) = 951 numbers.
    """
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += n_in * n_out + n_out
    return total


def random_genome(layer_sizes):
    """A brand-new random genome: every number between -1 and 1."""
    return np.random.uniform(-1, 1, genome_size(layer_sizes))


# ---- the network ----------------------------------------------------------
class NeuralNetwork:
    def __init__(self, layer_sizes, genome, hidden_activation='relu', output_activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.hidden_activation = ACTIVATIONS[hidden_activation]
        self.output_activation = ACTIVATIONS[output_activation]

        # Cut the genome into one weight matrix and one bias vector per layer.
        # The genome holds them in this order: W1, b1, W2, b2, ...
        self.weights = []
        self.biases = []
        pos = 0
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = genome[pos: pos + n_in * n_out].reshape(n_out, n_in)    # W[to, from]
            pos += n_in * n_out
            b = genome[pos: pos + n_out]
            pos += n_out
            self.weights.append(W)
            self.biases.append(b)

        # The values of every layer from the last forward() call. The dashboard draws these.
        self.activations = [np.zeros(n) for n in layer_sizes]

    def forward(self, inputs):
        """Feed the inputs through every layer and return the output layer."""
        values = np.array(inputs, dtype=float)
        self.activations = [values]
        last_layer = len(self.weights) - 1
        for i in range(len(self.weights)):
            values = self.weights[i] @ values + self.biases[i]
            if i == last_layer:
                values = self.output_activation(values)
            else:
                values = self.hidden_activation(values)
            self.activations.append(values)
        return values
