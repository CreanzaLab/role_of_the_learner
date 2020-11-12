import time
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import seaborn as sns


class CultureMatrix:
    def __init__(self, dim: tuple, culture_types: int,
                 proportion_novelty: float,
                 mort: float, ideal_neighbors: float,
                 learning_mutation_rate: float,
                 culture_mutation_rate: float,
                 seed=662, record_iter=5):
        self.iter = 0
        self.dim = dim
        self.mort = mort
        self.ideal_neighbors = ideal_neighbors
        self.learning_mutation_rate = learning_mutation_rate
        self.culture_mutation_rate = culture_mutation_rate
        np.random.seed(seed)
        self.seed = seed
        self.record_iter = record_iter
        self.entropy_history = []
        # self.overlap_history = []
        self.learning_history = np.empty((2, 0))
        self.culture_matrix = np.zeros(self.dim)
        self.learning_matrix = np.zeros(self.dim)
        self._initialize(culture_types, proportion_novelty)

    def _initialize(self, culture_types, proportion_novelty):
        # initialize matrices

        # initialize matrix of culture types, of size 'dim', with 'culture_types'
        self.culture_matrix = np.random.randint(
            1, 1 + culture_types, size=[*self.dim], dtype='int'
        )

        # initialize matrix of binary learning types, 0=conformity, 1=novelty
        self.learning_matrix = np.random.random([*self.dim])
        self.learning_matrix[
            self.learning_matrix > np.quantile(
                self.learning_matrix, proportion_novelty
            )
            ] = 0
        self.learning_matrix[
            self.learning_matrix != 0
            ] = 1
        self.learning_matrix = self.learning_matrix.astype(int)

    def culture_counts(self):
        counts = np.unique(self.culture_matrix, return_counts=True)[1]
        return counts

    def learning_counts(self):
        learning_types, counts = np.unique(self.learning_matrix, return_counts=True)

        if len(counts) == 1:
            if learning_types == [0]:
                counts = [*counts, 0]
            else:
                counts = [0, *counts]

        return counts

    def time_step(self):
        self._death()
        self._replacement()
        self._learning()

        if self.iter % self.record_iter == 0:
            self.learning_history = np.append(
                self.learning_history,
                np.reshape(self.learning_counts(),
                           (2, 1)), axis=1)
            self.entropy_history.append(self.entropy())
            # self.overlap_history.append(self.overlap())
        self.iter += 1

    def _death(self):
        # Every individual has an equal mortality rate.

        # Apply deaths mask:
        self.culture_matrix[
            np.random.binomial(1, self.mort, self.dim) == 1
            ] = 0

    def _replacement(self):
        # Replace 'dead' individuals with a method of learning,
        #   taken from living neighbors. Individuals singing with
        #   cultures closer to proportion to the 'ideal' value
        #   are favored. If no living neighbor exists, pick at
        #   random from the population.

        # For learning and culture matrices, create a padded matrix
        #   surrounded by "dead" dummies, so that the matrix does
        #   not wrap due to indexing.
        culture_padded = np.pad(self.culture_matrix, 1, constant_values=0)
        learning_padded = np.pad(self.learning_matrix, 1, constant_values=(-1))
        learning_padded[culture_padded == 0] = -1

        # Create a mutation list--booleans representing whether each new
        #   individual gets a random learning type.
        mutates = list(
            np.random.random(
                np.sum(self.culture_matrix == 0)
            ) < self.learning_mutation_rate
        )

        max_diff = max(abs(self.ideal_neighbors),
                       abs(1 - self.ideal_neighbors))

        # Loop through the individuals that need to be replaced,
        #   and choose a random living neighbor to replace them.
        enumerated = [n for n in np.ndenumerate(self.culture_matrix)]
        np.random.shuffle(enumerated)
        for idx, replace in enumerated:
            if replace != 0:
                # ignore values that do not need to be replaced
                pass
            else:
                # find a replacement for locations that need to be replaced
                if mutates.pop():
                    # if there is a mutation, randomly select between a conformity
                    #   and novelty preference.
                    self.learning_matrix[idx] = np.random.choice([0, 1])
                else:
                    select_from = (learning_padded[
                                   idx[0]:idx[0] + 3,
                                   idx[1]:idx[1] + 3]).flatten()
                    select_from = select_from[select_from != -1]
                    # Is there a living neighbor?
                    if len(select_from) != 0:
                        # If there are living neighbors, choose based on
                        #   the proportion of culture types in the vicinity
                        if len(np.unique(select_from)) == 1:
                            # Special case: if only one type of neighbor
                            self.learning_matrix[idx] = np.unique(select_from)
                        else:
                            # Selection uses neighboring culture types.
                            select_using = (culture_padded[
                                            idx[0]:idx[0] + 3,
                                            idx[1]:idx[1] + 3]).flatten()
                            select_using = select_using[select_using != 0]
                            if len(np.unique(select_using)) == 1:
                                # If there is only a single neighboring culture
                                #   type, then choose at random.
                                self.learning_matrix[idx] = np.random.choice(select_from)
                            else:
                                cultures, weights = np.unique(select_using, return_counts=True)
                                weights = max_diff - abs(weights / sum(weights) - self.ideal_neighbors)
                                weights = weights / sum(weights)
                                culture_selected = np.random.choice(cultures, p=weights)
                                self.learning_matrix[idx] = np.random.choice(
                                    select_from[select_using == culture_selected])

                    else:
                        # If not, choose at random from the population.
                        self.learning_matrix[idx] = np.random.choice(
                            learning_padded[learning_padded != -1]
                        )

    def _learning(self, learning_radius: int = 1):
        # Have newly replaced individuals learn according to
        #   the method each has adopted. Similarly, try to learn
        #   from neighbors, otherwise learn at random.
        # learning_radius: size of window in which to search for
        #   neighboring cultures.

        # Create a padded matrix surrounded by "dead" dummies,
        #   so that the matrix does not wrap due to indexing.
        culture_padded = np.copy(self.culture_matrix)
        culture_padded = np.pad(culture_padded, learning_radius, constant_values=0)

        # Create a mutation list, booleans representing whether each new
        #   individual has a new culture type.
        mutates = list(
            np.random.random(
                np.sum(self.culture_matrix == 0)
            ) < self.culture_mutation_rate
        )

        # Loop through the individuals that need to be replaced,
        #   and if there is no mutation, choose a random living
        #   neighbor's culture learn.
        enumerated = [n for n in np.ndenumerate(self.culture_matrix)]
        np.random.shuffle(enumerated)
        for idx, replace in enumerated:
            if replace != 0:
                # ignore values that do not need to be replaced
                pass
            else:
                if mutates.pop():
                    # If there is a mutation, learn a completely novel culture type
                    self.culture_matrix[idx] = self.culture_matrix.max() + 1
                else:
                    select_from = (culture_padded[
                                   idx[0]:idx[0] + 1 + 2 * learning_radius,
                                   idx[1]:idx[1] + 1 + 2 * learning_radius]).flatten()
                    select_from, counts = np.unique(
                        select_from[select_from != 0], return_counts=True)
                    counts = counts.astype(np.float)
                    # is there a living neighbor?
                    if len(select_from) != 0:
                        if self.learning_matrix[idx] == 0:
                            # if conformity-seeking
                            self.culture_matrix[idx] = np.random.choice(
                                select_from, p=(counts ** 2) / np.sum(counts ** 2))
                        else:
                            # if novelty-seeking
                            self.culture_matrix[idx] = np.random.choice(
                                select_from, p=(counts ** -2) / np.sum(counts ** -2))
                    else:
                        # if no living neighbor, choose at random
                        self.culture_matrix[idx] = np.random.choice(
                            culture_padded[culture_padded != 0]
                        )

    def run(self, iterations: int):
        # Start the process, running it for some number of time-steps
        for _ in range(iterations):
            self.time_step()

    def plot(self, bin_size=5):
        # Create plots to visualize the data
        culture_counts = self.culture_counts()
        bin_count = np.bincount(culture_counts)
        count_binned = [np.sum(bin_count[i:i + bin_size]) for i in
                        range(0, len(bin_count), bin_size)]

        x = np.arange(len(count_binned)) * bin_size
        y = count_binned / np.sum(count_binned)
        df = pd.DataFrame({'counts': y, 'bins': x})

        my_dpi = 96
        sns.set(style='white')
        sns.set_style('ticks')
        sns.set_context({"figure.figsize": (20, 7)})
        plt.figure()

        ax = df.plot(x='bins',
                     y='counts',
                     kind='bar',
                     use_index=True,
                     grid=None,
                     rot=0,
                     width=0.95,
                     fontsize=10,
                     linewidth=0,
                     color='k')

        plt.title("sfs: ")
        plt.xlabel("num. indiv's with culture types")
        plt.ylabel("num. of culture types")

        plt.tight_layout()

    def entropy(self):
        # To estimate the patchiness of the culture matrix,
        #  this function finds the entropy of the magnitude
        #  of the Sobel discrete differentiation operator.
        # (idea thanks to https://stats.stackexchange.com/a/250319)

        # Calculate the Sobel magnitudes for each extant
        #  culture-type found in the culture matrix.

        # The x- and y- direction kernels for convolution:
        gx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        gy = gx.transpose()

        sobel_magnitudes = np.zeros_like(self.culture_matrix).astype(np.float)

        # Sum Sobel magnitudes calculated for each culture type
        for cult_type in np.unique(self.culture_matrix):
            sobel_magnitudes += (
                                        convolve(self.culture_matrix == cult_type,
                                                 gx, mode='nearest') ** 2 +
                                        convolve(self.culture_matrix == cult_type,
                                                 gy, mode='nearest') ** 2
                                ) ** 0.5

        # Find the Shannon entropy of the magnitudes:
        return entropy(
            sobel_magnitudes.flatten()
        )

    def visualize_matrix(self, ax=None):
        # Calculate the Sobel magnitudes for each extant
        #  culture-type found in the culture matrix.

        # The x- and y- direction kernels for convolution:
        gx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        gy = gx.transpose()

        sobel_magnitudes = np.zeros_like(self.culture_matrix).astype(np.float)

        # Sum Sobel magnitudes calculated for each culture type
        for cult_type in np.unique(self.culture_matrix):
            sobel_magnitudes += (
                convolve(self.culture_matrix == cult_type,
                         gx, mode='nearest') ** 2 +
                convolve(self.culture_matrix == cult_type,
                         gy, mode='nearest') ** 2
                                ) ** 0.5

        if ax is None:
            plt.imshow(sobel_magnitudes, cmap="viridis")
        else:
            ax.imshow(sobel_magnitudes, cmap="viridis")


# Written reusing some code written previously by Abigail Searfoss (
#   https://github.com/CreanzaLab/chipping_sparrow_cultural_transmission_model)

# How many time-steps should the model be run before we see the results?
ITERATIONS = 4000
# How large should the square matrix be?
DIM = (128, 128)
# Set seed
SEED = 662
np.random.seed(SEED)
# What proportion of individuals should be replaced in each time-step?
MORTALITY_RATE = 0.2
# How often should novelty or conformity preference randomly appear in a new individual?
LEARNING_MUTATION_RATE = 0.0001
# How often should an individual learn a completely novel cultural type?
CULTURE_MUTATION_RATE = 0.005
# How many cultural types should exist at the beginning?
CULTURE_TYPES = 16
# What proportion of the starting population should have a novelty preference?
PROPORTION_NOVELTY = 0.25
# How many times should this process be replicated?
REPLICATES = 5
# should the figures be saved to pdf & png
savefigs = True

# setup for the models:

# test models with the following ideal-neighbor-proportions:
ideal_neighbor_to_test = []
[ideal_neighbor_to_test.append(i/10) for i in range(5)]
[ideal_neighbor_to_test.append(0.5 + i/20) for i in range(11)]
ideal_neighbor_to_test = np.repeat(ideal_neighbor_to_test, REPLICATES)

# create models to test
culture_matrix_list = [
    CultureMatrix(DIM, CULTURE_TYPES, PROPORTION_NOVELTY,
                  MORTALITY_RATE,
                  ideal_neighbors=inp,
                  learning_mutation_rate=LEARNING_MUTATION_RATE,
                  culture_mutation_rate=CULTURE_MUTATION_RATE,
                  seed=np.random.randint(662000))
    for inp in ideal_neighbor_to_test
]

# run the models:

for i in range(len(culture_matrix_list)):
    print("Run: {}/{}   ideal_neighbors: {}  seed: {}".format(
        i + 1, len(culture_matrix_list),
        culture_matrix_list[i].ideal_neighbors,
        culture_matrix_list[i].seed))
    print(time.asctime())
    culture_matrix_list[i].run(ITERATIONS)

# review results:

# plot example of final culture frequencies
culture_matrix_list[0].plot(bin_size=10)
plt.show()


# plot final novelty values per INP
def plot_results_novelty(results_list):
    # create scatter-plot of final conformity results for all replicates and ideal-neighbor-proportions
    x = np.array([n.ideal_neighbors for n in results_list])
    y = np.array([np.mean(n.learning_history[1, -10:]) /
                  np.sum(n.learning_history[:, 0]) for n in results_list])
    y_err = np.array([np.std(y[x == x_val]) for x_val in np.unique(x)]) * np.sqrt(REPLICATES / (REPLICATES - 1))

    plt.scatter(x, y, color="k")
    plt.xlim((-0.05, 1.05))
    plt.ylim((0, 1.1))
    plt.scatter(x[0:len(x):REPLICATES],
                [np.mean(y[i*REPLICATES:(i+1)*REPLICATES]) for i in range(int(len(x)/REPLICATES))],
                s=20, color='r', marker='_')
    plt.errorbar(x[0:len(x):REPLICATES],
                 [np.mean(y[i*REPLICATES:(i+1)*REPLICATES]) for i in range(int(len(x)/REPLICATES))],
                 yerr=y_err, fmt='none', color="red", elinewidth=3)
    plt.axhline(1, 0, 1, ls="--", c='k')
    plt.xlabel("Ideal neighbor proportion")
    plt.ylabel("Proportion Novelty")


f1 = plt.figure(figsize=(15, 9))
plot_results_novelty(culture_matrix_list)
if savefigs:
    f1.savefig("Final_conformity_fraction.pdf", bbox_inches='tight')
    f1.savefig("Final_conformity_fraction.png", bbox_inches='tight')
plt.show()

# plot proportion novelty-preference over time
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color",
    plt.cm.viridis(np.linspace(0, 1, int(len(culture_matrix_list)/REPLICATES))))

f2 = plt.figure(figsize=(20, 10))

for ideal_n in np.unique([n.ideal_neighbors for n in culture_matrix_list]):
    plt.plot([i*5 for i in range(int(ITERATIONS/5))],
             np.transpose(
                 np.mean([n.learning_history[1, :]/np.sum(n.learning_history[:, 0])
                          for n in culture_matrix_list if n.ideal_neighbors == ideal_n],
                         axis=0)),
             label=str(ideal_n)
             )

plt.axhline(1, 0, 5000, ls="--", c='k')
plt.legend(title="INP", loc="upper left")
plt.xlabel("Time-step")
plt.ylim((0.15, 1.1))
plt.xlim((ITERATIONS * -0.1, ITERATIONS * 1.1))
plt.ylabel("Proportion novelty preference")

if savefigs:
    f2.savefig("Novelty_over_time.pdf", bbox_inches='tight')
    f2.savefig("Novelty_over_time.png", bbox_inches='tight')
plt.show()

# plot 'entropy' over time
f3 = plt.figure(figsize=(20, 10))

for ideal_n in np.unique([n.ideal_neighbors for n in culture_matrix_list]):
    plt.plot([i*5 for i in range(int(ITERATIONS/5))],
             np.transpose(
                 np.mean([n.entropy_history for n in culture_matrix_list if n.ideal_neighbors == ideal_n],
                         axis=0)),
             label=str(ideal_n)
             )
plt.legend(title="INP", loc="lower left")
plt.xlim((0, ITERATIONS))
plt.xlabel("Time-step")
plt.ylabel("Entropy")

if savefigs:
    f3.savefig("Entropy_over_time.pdf", bbox_inches='tight')
    f3.savefig("Entropy_over_time.pnf", bbox_inches='tight')
plt.show()

# examples of entropy plots for each INP
f4 = plt.Figure(figsize=(20, 20))

for subplot, i in enumerate(np.arange(4, len(ideal_neighbor_to_test), step=REPLICATES)):
    ax = f4.add_subplot(4, 4, subplot+1)
    ax.set_title("INP: " + str(ideal_neighbor_to_test[i]))
    culture_matrix_list[i].visualize_matrix(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

if savefigs:
    f4.savefig("example_INP_entropy.pdf", bbox_inches='tight')
    f4.savefig("example_INP_entropy.png", bbox_inches='tight')
plt.show()

