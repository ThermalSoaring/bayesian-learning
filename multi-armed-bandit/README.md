# Multi-Armed Bandit Problem
This is an attempt at applying Bayesian learning to the [Multi-armed
Bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem.

## Python Implementation
#### bayesian-bandit.ipynb
This is the Jupyter Notebook of the successful implementation in Python
using PyMC3. Also available as a PDF or .py file.

## Sage Attempt
#### wilson-bayes-seminar-complicated-temperature-example.sage
Starting point just recreating a graph out of a
[presentation](https://www.cs.tcd.ie/disciplines/statistics/statica/statica_web/Wilson_Bayes_seminar.pdf)

#### bayesian-learning-demo.sage
A demo trying to learn normal distribution with Bayes' theorem  

#### bayesian-learning-eps.sage
Trying to still use the Epsilon-greedy strategy but use Bayes' theorm to
learn the mean reward for pulling a lever. Not working.

#### bayesian-learning-interval-estimation.sage
Trying to apply [interval
estimation](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node25.html)
instead of the Epsilon-greedy strategy. Not working.
