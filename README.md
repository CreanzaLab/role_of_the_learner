# role_of_the_learner
A spatial model of cultural evolution

## running
Written in Python 3.8. Requires NumPy, SciPy, Pandas, Matplotlib, and Seaborn*.

*: Seaborn not strictly necessary; the plot() function can be removed from the model with no side effects.

Generates a matrix of individuals. Each timestep, individuals die, are replaced, and learn based on their inherited preferences. 
