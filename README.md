# role_of_the_learner
A spatial model of cultural evolution

## running
Written in Python 3.8. Requires NumPy, SciPy, Pandas, Matplotlib, and Seaborn*.

*: Seaborn not strictly necessary; the plot() function can be removed from the model with no side effects.

This code models the evolution of a population's learning preferences and culture over time. A matrix of individuals, each with a single culture type is generated at the beginning of each simulation. Every timestep, some will randomly die. Living neighbors are then selected based on the probability of their culture type in the local area, and pass on their learning preference. These new individuals learn from their neighbors based on their inherited learning preferences, and the timestep finishes.

Each simulation file contains the following variables, which can be changed to affect the model functioning:

`ITERATIONS (default: 4000):`  Each simulation is run for this many timesteps

`DIM (default: 128*128):`  the length and width of the population matrix; 128\*128 = 16,384 individuals

`SEED (default: 662):`  the initial for random number generation. This is updated for each simulation

`MORTALITY_RATE (default: 0.2):`  the proportion of individuals that are replaced in each timestep

`LEARNING_MUTATION_RATE (default: 0.0001):`  how often a new individual is randomly assigned a novelty or conformity preference

`CULTURE_MUTATION_RATE (default: 0.005):`  how often a new individual is randomly assigned a completely new culture type

`CULTURE_TYPES (default: 16):`  how many initial culture types exist

`PROPORTION_NOVELTY (default: 0.25):`  proportion of the starting population that has a novelty preference

`REPLICATES (default: 5):`  number of simulations run for each variable setting

`RECORD_ITER (default: 5):`  frequency, in timesteps, of recording information about the population for plotting

`savefigs (default: True):`  whther the figures should be saved to pdf & png (provide an `out` directory or edit the output names to ensure correct saving)
