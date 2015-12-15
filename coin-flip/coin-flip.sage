#
# Bayesian coin-flip learning
#
# We have a coin that is weighted a particular way and want to learn using
# Bayes' theorem what that weighting is given some evidence.
#
# Note: Based on p. 185-188 of Bayesian Artificial Intelligence,
#       2nd Ed. by Korb and Nicholson

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

# Updating rule
#
#    P(theta|heads) = beta*P(heads|theta)*P(theta)
#        where beta is the inverse of the probability of the evidence,
#        i.e., a factor for the denominator of Bayes' theorem
# => P(theta|heads) = beta*theta*P(theta)
#        since P(heads|Theta=theta)=theta
#
# And generalizing, we get:
#
#    P(theta|e) = beta*theta^m*(1-theta)^(n-m)*P(theta)
#        where e is the evidence, m # of heads, and n-m # of tails,
#        assuming coin tosses are independent and identically distributed (iid)
#
# Or, we can use the beta distribution B(alpha1, alpha2)
#
#    P(theta|alpha1,alpha2) =
#        beta*theta^(alpha1-1)*(1-theta)^(alpha2-1)
#
# So with evidence we will have:
#
#    P(theta|e,alpha1,alpha2) =
#        beta*theta(alpha1+m-1)*(1-theta)^(alpha2+(n-m)-1)
#
# P_theta -- posterior probability density function given m and n
#            ignoring the normalization factor
#     where n = total number
#           m = number of heads
P_theta(theta, alpha1, alpha2, m, n) = theta^(alpha1+m-1)*(1-theta)^(alpha2+(n-m)-1)

# Running the experiment
#
# Weighting of the coin:
#   weighting - 0 is 100% tails, 1 is 100% heads
#
# Hyperparameters:
#   alpha1 - heads, represented by 1
#   alpha2 - tails, represented by 0
#
# Experiment factors:
#   Iterations - how many times to flip the coin
#   Curves - how many curves you want displayed on the graph
#
# The "equivalent sample size"
#   The larger alpha1+alpha2 is, the longer it will take to shift the learned
#   distribution away from our prior. If these are the same, our prior belief
#   is that the weighting is 0.5.
def experiment(weighting, alpha1, alpha2, iterations, curves):
    # List of plots
    plots = []
    #cols = ['#FF0000','#0000FF','#FFFF00','#00FF00'] # r,b,y,g
    cols = rainbow(curves)

    # Initial values
    n = 0
    m = 0

    for i in range(0, iterations):
        # Flip a coin using the weighting to choose the chance of flipping
        # heads or tails
        flipValue = weighted_choice([(0,1-weighting),(1,weighting)])

        # Increment total flip count
        n += 1

        # If we flipped heads, increment head count
        if flipValue == 1:
            m += 1

        # Compute the new probability distribution
        unnormalized(theta) = P_theta(theta, alpha1, alpha2, m, n)
        norm_factor = integral(unnormalized, theta, 0, 1)
        posterior(theta) = unnormalized/norm_factor

        # Only display the desired number of curves on the graph
        modFactor = ceil(iterations/curves)

        if i%modFactor == 0:
            plots.append(plot(posterior,theta,xmin=0,xmax=1,
                    color=cols[i/modFactor],legend_label='B('+str(m)+','+str(n-m)+')',
                    axes_labels=['$\\theta$','P($\\theta$|e)'],
                    title='Belief about coin weighting given $\\theta = '+ '%.2f'%weighting +'$, '
                    +'$\\alpha_1 = '+str(alpha1)+', '
                    +'\\alpha_2 = '+str(alpha2)+'$'))

    return sum(plots)

# Experiments
iterations = 100
curves = 7

# Illustrates how it'll learn different values for different actual weightings
weight_change = [
    experiment(0.2, 5, 5, iterations, curves),
    experiment(0.7, 5, 5, iterations, curves)]

for i in range(len(weight_change)):
    weight_change[i].save('weight_change_'+str(i)+'.png')

# Illustrates how increasing alpha1+alpha2 makes it take longer to learn a
# value based on the evidence
alpha_change = [
    experiment(0.2, 1, 1, iterations, curves),
    experiment(0.2, 5, 5, iterations, curves),
    experiment(0.2, 50, 50, iterations, curves)]

for i in range(len(alpha_change)):
    alpha_change[i].save('alpha_change_'+str(i)+'.png')
