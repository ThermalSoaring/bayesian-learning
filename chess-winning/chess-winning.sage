#
# Bayesian learning a multinomial distribution
#
# Two individuals play a chess game. We specify the probability that player A
# wins, player B wins, or that the game ends in a draw. Thus, there are three
# possible outcomes. We want to learn what these probabilities are through
# watching the players play a number of games.
#
# Note: Based on p. 188-190 of Bayesian Artificial Intelligence,
#           2nd Ed. by Korb and Nicholson
#       with the example multinomial problem taken from
#       http://onlinestatbook.com/2/probability/multinomial.html

import random

# Choose items in a list randomly based on a certain chance of being selected
#
# Example:
#    [('a',1.0),('b',2.0),('c',3.0)]
#
# Source: http://stackoverflow.com/a/3679747
def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

# Running the experiment
#
# actual -- The actual probabilities of the possible outcomes.
# alphas -- The hyperparameters, reflect prior belief and degree
#     of confidence in this belief
# iterations -- How many times to run the experiment.
# curves -- How many curves to put on the graph.
#
def experiment(actual, alphas, iterations, curves):
    assert len(actual) == len(alphas), \
        "lengths of actual and alphas must be equal"
    assert iterations > 0 and curves > 0, \
        "iterations and curves must be greater than zero"

    # The likelihood of getting each of the possible outcomes
    weightedChoices = [(i, actual[i]) for i in range(len(alphas))]

    # How many times we've gotten each outcome
    outcomeCount = [0 for i in range(len(alphas))]

    for i in range(0, iterations):
        # Choose randomly based on the weighting
        outcome = weighted_choice(weightedChoices)

        # Increment the number of times we've had this outcome
        outcomeCount[outcome] += 1

        # Compute the new probability distribution
        hyperparameter_total = [alphas[j]+outcomeCount[j] for j in range(len(alphas))]
        hyperparameter_sum = sum(hyperparameter_total)
        posterior = [hyperparameter_total[j] / hyperparameter_sum for j in range(len(alphas))]

        # Only display the desired number of curves on the graph
        modFactor = ceil(iterations/curves)

        if i%modFactor == 0:
            print "Outcomes:", outcomeCount, "P:", posterior

        # For the last one, round it so we can compare with the correct probabilities
        if i == iterations-1:
            print
            print "Correct: ",[N(j) for j in actual]
            print "Measured: ",[N(j) for j in posterior]

# Experiments
#
# P(Player A wins) = 0.4
# P(Player B wins) = 0.35
# P(Draw) = 0.25
#
experiment([0.4, 0.35, 0.25], [1,1,1], 10000, 10)
