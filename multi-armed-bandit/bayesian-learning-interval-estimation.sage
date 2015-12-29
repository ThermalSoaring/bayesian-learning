#
# Applying Bayesian Learning to The Multi-Armed Bandit Problem
#
# Garrett Wilson
# November 1, 2015
#
# The goal of this simulation is to apply Bayesian learning to learning which
# levers to pull to get the most money.
#
# Let the random variables X_1, X_2, ... X_n be the amount of money you get
# each time you pull a particular lever. We will assume that these follow the
# Markov property with the new probability distribution only depending on the
# last probability distribution rather than all those before, i.e. X_n depends
# only on X_(n-1) rather than X_1, X_2, ... X_(n-1).
#
# We want this to be true since it'll simplify calculations. We will consider
# the simulation a success if it improves upon the results from applying the
# epsilon-greedy algorithm to the problem.
#
# We will assume that the amount of money gained per pull of a certain lever is
# a normal distribution with a certain standard deviation and mean. We want to
# learn this standard deviation but in particular the mean. Then we will pull
# the levers with the higher mean more often while at the same time becoming
# more confident about where that mean is.
#
# We will assume an initial uniform distribution from [0,max] where max is some
# reasonable maximum for how much each lever pull will give you, depends on
# currency unit.
#
# Note: We will also for the time being assume we know the standard deviation.
#
#
# What I learned:
#
#    I'm not really improving the results any. The Bayesian learning learns the
#    probability distribution. This doesn't tell us how we should explore the
#    levers though, so we still have to then use some decision-making algorithm
#    to determine what to do with these learned probability distributions.
#
#    For example, maybe classification really *is* what we need here. We want
#    to determine which of the levers is the best to pull based on our
#    distributions.
#
# Used the following slides to gain an understanding of how to do this:
# https://www.cs.tcd.ie/disciplines/statistics/statica/statica_web/Wilson_Bayes_seminar.pdf
#
import re
import mpmath

# Find all inverse_erf and evaluate using mpmath
def eval_inverse_erf(obj):
    s = str(obj)
    groups = re.search('inverse_erf\(([^\)]*)\)', s).groups()

    for i in groups:
        val = RR(mpmath.erfinv(i))
        s = re.sub('inverse_erf\('+re.escape(i)+'\)', str(val), s)

    return sage_eval(s)

def GenerateLevers(numLev):
    # Returns the mean (normlized to (0,1)) reward of the levers.
    # There are numLev levers.
    levers = [RR.random_element(0,1) for i in range(0,numLev)]
    return levers

def LeverReward(leverMeanReward):
    # Standard deviation
    stdev = 0.1
    reward = RealDistribution('gaussian', stdev).get_random_element()+leverMeanReward
    return reward

def GreedyEps(epsilon, numIter, leverReal):
    # Using the greedy epsilon methods, calculates reward at each iteration
    # Returns a vector, given reward for each round

    # leverReal =  Actual mean reward per leaver
    numLevers = len(leverReal)

    # Estimate the mean reward of each lever
    leverRewEst = [0 for k in range(0, numLevers)] # Average reward received per lever
    totalReward = [0 for k in range(0, numLevers)] # How much reward received per lever
    numPullsDone = [0 for k in range(0, numLevers)] # Pulls per lever

    # Actual reward received
    reward = [0 for k in range(0, numIter)]

    # Pull a lever and find reward for it numIter times
    for i in range(0, numIter):
        # Pull a lever
        # Choose to explore or pull most greedy
        
        # Choose to pull the most greedy lever
        # (if epsilon = 0 is always greedy)
        curiosity = RR.random_element(0,1)

        if curiosity >= epsilon: # Be greedy
            m = max(leverRewEst)
            maximums = [k for k, j in enumerate(leverRewEst) if j == m]
            leverPulled = maximums[0]
        else: # Explore
            leverPulled = ZZ.random_element(0, numLevers)
            
        # Record reward from the lever chosen
        meanLeverChosen = leverReal[leverPulled]
        reward[i] = LeverReward(meanLeverChosen)
        totalReward[leverPulled] += reward[i]
       
        # Update estimate of lever reward
        numPullsDone[leverPulled]+=1
        leverRewEst[leverPulled] = totalReward[leverPulled]/numPullsDone[leverPulled]

    return reward

def BayesianLearning(numIter, leverReal):
    # Using the Bayesian learning methods, calculates reward at each iteration
    # Returns a vector, given reward for each round

    # leverReal =  Actual mean reward per leaver
    numLevers = len(leverReal)

    # Estimate the mean reward of each lever
    leverRewEst = [0 for k in range(0, numLevers)] # Average reward received per lever
    totalReward = [0 for k in range(0, numLevers)] # How much reward received per lever
    numPullsDone = [0 for k in range(0, numLevers)] # Pulls per lever
    distributions = [0 for k in range(0, numLevers)] # Probability distributions

    # 95% upper confidence interval
    # Initialize to 1 since we want to initially look at all the levers. This
    # may indicate that this algorithm is flawed.
    confidence95 = [1 for k in range(0, numLevers)]

    # Actual reward received
    reward = [0 for k in range(0, numIter)]

    # Start out assuming we know sigma
    var('t')
    normalDist(t,x,sigma)=1/sqrt(2*pi*sigma^2)*e^(-1/2*(x-t)^2/sigma^2)
    known_sigma=1/10
    p(t,x)=normalDist(t,x,sigma=known_sigma)

    # Pull a lever and find reward for it numIter times
    for i in range(0, numIter):
        m = max(confidence95)
        maximums = [k for k, j in enumerate(confidence95) if j == m]
        leverPulled = maximums[0]

        # Record reward from the lever chosen
        meanLeverChosen = leverReal[leverPulled]
        reward[i] = QQ(LeverReward(meanLeverChosen)) # must be exact... Sage numerical issues
        totalReward[leverPulled] += reward[i]
       
        # Update estimate of lever reward
        numPullsDone[leverPulled]+=1
        #leverRewEst[leverPulled] = totalReward[leverPulled]/numPullsDone[leverPulled]

        # Initially just use the measurement as the mean, using a uniform
        # distribution with limits of integration just results in the
        # normalization factor having erf() in it and evaluating to about zero
        # giving us divide by zero issues.
        if distributions[leverPulled] == 0:
            post(t)=p(t,x=reward[i])

        # Otherwise, our prior is the last iteration's posterior
        else:
            eq(t)=distributions[leverPulled]*p(t,x=reward[i])
            norm_factor=integral(eq,t,-oo,oo)
            post(t)=eq(t)/norm_factor

        # Update probability distribution
        distributions[leverPulled]=post

        # Update mean amount gained for each lever based on present probability
        # distributions
        leverRewEst[leverPulled] = solve(diff(distributions[leverPulled],t)==0,t)[0].right()
        # Update 95 percent upper confidence interval
        F(x) = integral(distributions[leverPulled],t,x,oo)
        confidence95[leverPulled] = N(eval_inverse_erf(solve(F(t)==95/100,t)[0].right()))

    return reward

def Framework():
    # Plots the reward per pull as a function of # of tries.
    # Does many trials to smooth out results
    # Tries different epsilons or greediness levels
    greedyAvgReward = []
    greedyMaxReward = []
    epsilon = [0, 1/100, 1/10, 1/2]
    cols = ['#FF0000','#0000FF','#FFFF00','#00FF00'] # r,b,y,g

    numPullsPerTrial = 30
    numTrials = 50
    #numPullsPerTrial = 300
    #numTrials = 500
    numLevers = 10

    # Epsilon greedy approach
    #for e in range(0, len(epsilon)):
    if False:
        rewards = []
        bestLevers = []

        for t in range(0, numTrials):
            levers = GenerateLevers(numLevers)
            bestLevers.append(max(levers))
            rewards.append(GreedyEps(epsilon[e], numPullsPerTrial, levers))

        # Average the rows to get the average reward at various times
        # To do this, we take the average of each column
        #avgReward = [mean(r) for r in rewards] # avg rows
        avgReward = [mean([r[i] for r in rewards]) for i in range(0,numPullsPerTrial)] # avg cols
        avgBest = mean(bestLevers)

        # Plot the average reward across many trials
        greedyAvgReward.append(list_plot([k/avgBest for k in avgReward],color=cols[e],
            legend_label='$epsilon='+str(epsilon[e])+'$',
            title='Performance of Epsilon Greedy Algorithms as Epsilon is Varied',
            axes_labels=['Lever Pulls','Avg. Reward']))

        # Plot the maximum reward possible
        #bestLever = avgBest/avgBest
        #greedyMaxReward.append(plot([0 numPullsPerTrial], [1 1]*bestLever,'-r' ))

    rewards = []
    bestLevers = []

    for t in range(0, numTrials):
        levers = GenerateLevers(numLevers)
        bestLevers.append(max(levers))
        rewards.append(BayesianLearning(numPullsPerTrial, levers))
        print "Done with Bayesian trial", t

    # Average the rows to get the average reward at various times
    # To do this, we take the average of each column
    avgReward = [mean([r[i] for r in rewards]) for i in range(0,numPullsPerTrial)] # avg cols
    avgBest = mean(bestLevers)

    # Plot the average reward across many trials
    bayesianPlot=list_plot([k/avgBest for k in avgReward],color=cols[e],
        legend_label='Bay.', axes_labels=['Lever Pulls','Avg. Reward'])

    #show(sum(greedyAvgReward))
    show(bayesianPlot)

Framework()
