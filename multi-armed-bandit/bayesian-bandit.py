#
# Bayesian Bandit
#
# Based on:
#  David's implementation of the Epsilon-Greedy algorithm
#    https://github.com/ThermalSoaring/RL-Multi-Armed-Bandit
#  Probabilistic Programming and Bayesian Methods for Hackers, Chapter 6
#    http://nbviewer.ipython.org/github/CamDavidsonPilon/
#    Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
#    blob/master/Chapter6_Priorities/Chapter6.ipynb
#

# TODO: do this with a normal approximation rather than
# sampling since sampling is *incredibly* slow, see:
# Appendix of Chapter 4

import numpy as np
import pymc3 as pm
import scipy as sp
import matplotlib.pyplot as plt

# Display plots in Jupyter
#get_ipython().magic('matplotlib inline')

# For reproducibility
np.random.seed(123)

# The arm simulation model, where you can pull an arm and see how much
# money you get
class Arm:
    # Constructor: Arm(1,1)
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    # Calling: Arm(1,1)()
    def __call__(self):
        return np.random.normal(self.mean, self.stdev)

    # Printing: print(Arm(1,1))
    def __repr__(self):
        return '<%s.%s object at %s with mean=%s, stdev=%s>'% (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self)), self.mean, self.stdev)

# Using the same model for each subsequent pull of a lever drastically
# speeds up the sampling
class ArmModel:
    def __init__(self):
        self.model = pm.Model()

        with self.model:
            # We don't know where the mean is
            self.mu = pm.Uniform('mu', 0, 200)

            # Probably a small standard deviation
            #   lambda = 0.01 doesn't start out with too small of a stdev
            #   testval makes it so we don't get an error
            self.sd = pm.Exponential('sd', 0.01, testval=5)

            # The distribution with our observations
            self.arm = pm.Normal('arm', mu=self.mu, sd=self.sd)

class Simulation:
    def __init__(self, bandits, mean_low, mean_high,
                 stdev_low, stdev_high, figsize=(15, 4)):

        assert bandits>0, "# of bandits must be > 0"

        # Create the arms for the simulation
        self.arms = np.zeros(bandits, dtype=object)

        for i in range(0,bandits):
            mean = np.random.randint(mean_low, mean_high)
            stdev = np.random.randint(stdev_low, stdev_high)
            self.arms[i] = Arm(mean, stdev)

        # For consistently-sized plots
        self.figsize = figsize

    # Example: plotArms() or plotArms(bandits=[0,1,5,10])
    def plotArms(self, bandits=None, stdevs=5, bins=75, points=10000):
        plt.figure(figsize=self.figsize)
        plt.suptitle('Bandit Monetary Distributions')
        plt.xlabel('Monetary value on pulling lever')
        plt.ylabel('Probability of this value')
        plt.grid()

        # If list of bandits to display not specified, just use all of them
        if bandits == None:
            bandits = range(0, len(self.arms))

        for i in range(0,len(bandits)):
            samples = [self.arms[bandits[i]]() for j in range(0, points)]
            x = np.linspace(self.arms[bandits[i]].mean
                            - stdevs*self.arms[bandits[i]].stdev,
                            self.arms[bandits[i]].mean
                            + stdevs*self.arms[bandits[i]].stdev)
            plt.hist(samples, bins=bins, normed=True, histtype="stepfilled",
                     alpha=0.2, label='Bandit #'+str(bandits[i]))

        plt.legend()

    # Randomly pull levers and see how well we do
    def plotRand(self, trials=50, pullsPerTrial=300):
        plt.figure(figsize=self.figsize)
        plt.suptitle('Online "Learning" Algorithm: Random (a baseline)')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid()

        rewards = np.zeros((trials, pullsPerTrial))

        for trial in range(0, trials):
            for pull in range(0, pullsPerTrial):
                # Randomly select an arm to look at
                selected_arm = np.random.randint(0, len(self.arms))

                # Pull the arm and measure the reward
                rewards[trial, pull] = self.arms[selected_arm]()

        # Average the first pull of each trial, the second pull, etc.
        pullRewards = rewards.mean(axis=0)

        plt.plot(pullRewards)

    # Use the epsilon-greedy algorithm for determining when to exploit vs. explore
    def plotEpsGreedy(self, epsilons=[0.25,0.5,0.75,0.9], trials=500, pullsPerTrial=300):
        plt.figure(figsize=self.figsize)
        plt.suptitle('Online Learning Algorithm: Epsilon-Greedy')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid()

        for epsilon in epsilons:
            rewards = np.zeros((trials, pullsPerTrial))

            for trial in range(0, trials):
                # Which lever did we pull each time?
                selected_arm = np.zeros(pullsPerTrial, dtype=int)

                for pull in range(0, pullsPerTrial):
                    curiosity = np.random.random()

                    # Be greedy
                    if curiosity >= epsilon:
                        # Pull the lever with the most past reward
                        selected_arm[pull] = selected_arm[rewards[trial].argmax()]

                    # Explore, choose a random lever
                    else:
                        # Randomly select an arm to look at
                        selected_arm[pull] = np.random.randint(0, len(self.arms))

                    # Pull the arm and measure the reward
                    rewards[trial, pull] = self.arms[selected_arm[pull]]()

                    #if trial==0:
                    #    print(curiosity>=epsilon, selected_arm[pull],'of',
                    #          len(self.arms),'reward =',rewards[trial,pull])

            # Average the first pull of each trial, the second pull, etc.
            pullRewards = rewards.mean(axis=0)

            plt.plot(pullRewards, label='Epsilon='+str(epsilon))

        plt.legend()

    # The Bayesian Bandit solution
    def plotBayes(self, trials=10, pullsPerTrial=300):
        plt.figure(figsize=self.figsize)
        plt.suptitle('Online Learning Algorithm: Bayesian')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid()

        rewards = np.zeros((trials, pullsPerTrial))

        for trial in range(0, trials):
            # Which lever did we pull each time?
            selected_arm = np.empty(pullsPerTrial, dtype=int)
            selected_arm.fill(-1); # can't be a valid arm, e.g. can't be zero

            # Start with a uniform prior for each lever:
            #    (mu, stdev, 95% least plausible value)
            arm_beliefs = np.zeros([len(self.arms), 3], dtype=float)
            models = np.zeros(len(self.arms), dtype=object)

            for lever in range(0, len(self.arms)):
                arm_beliefs[lever] = (np.nan, np.nan, np.nan)
                models[lever] = ArmModel()

            # Start pulling levers!
            for pull in range(0, pullsPerTrial):
                nans = [np.isnan(mu) or np.isnan(sd) or np.isnan(lpv)
                        for mu, sd, lpv in arm_beliefs]

                # Start off by pulling each lever at least once so we have some data
                # to operate on
                if True in nans:
                    lever = nans.index(True)

                else:
                    # Select by highest 95% least plausible value (see Chapter 4)
                    lpv_min = arm_beliefs[:,2].min()
                    sq_diffs = np.array([(lpv - lpv_min)**2
                                         for mu, sd, lpv in arm_beliefs])
                    diff_sum = sq_diffs.sum()
                    p_by_lpv = [s / diff_sum for s in sq_diffs]

                    # Sample
                    lever  = np.random.choice(len(self.arms), 1,
                                              replace=False, p=p_by_lpv)[0]

                reward = self.arms[lever]()

                # Save results (must save before loading from rewards 2D array)
                selected_arm[pull] = lever
                rewards[trial, pull] = reward

                # What money have we gotten when we pulled this lever this time
                # and before?
                lever_observations = [rewards[trial, j]
                                      for j, arm in enumerate(selected_arm)
                                      if arm == lever]

                # Update information about this lever
                with models[lever].model:
                    # Don't update on every pull, too slow
                    if (len(lever_observations)-1)%5 == 0:
                        # Create normal distribution with observed values
                        models[lever].arm = pm.Normal('arm', mu=models[lever].mu,
                                             sd=models[lever].sd,
                                             observed=lever_observations)

                        # Sample this to find the posterior
                        step = pm.Metropolis(vars=[models[lever].mu,
                                                   models[lever].sd,
                                                   models[lever].arm])
                        trace = pm.sample(100, step=step, progressbar=False)

                        # Mean
                        arm_beliefs[lever][0] = trace[-1]['mu']
                        # Stdev
                        arm_beliefs[lever][1] = trace[-1]['sd']
                        # 95% least-plausible value
                        arm_beliefs[lever][2] = np.sort(trace['mu'])[int(
                                0.05 * len(trace['mu']))]

                    # Debugging at the end
                    if pull == pullsPerTrial-1:
                        print("Beliefs:")
                        for i, (mu, sd, lpv) in enumerate(arm_beliefs):
                            count = len([0 for j, arm in enumerate(selected_arm)
                                         if arm == i])
                            print("Count:", count, "Estimate: mu =", mu,
                                  "sd =", sd, "lpv =", lpv,
                                  "Actual: mu =", self.arms[i].mean,
                                  "sd =", self.arms[i].stdev)

        # Average the first pull of each trial, the second pull, etc.
        pullRewards = rewards.mean(axis=0)
        plt.plot(pullRewards, label='Bayes')
        plt.legend()

# bandits - How many bandits (or machines, or arms we'll pull) do we want?
# iterations - How many times will we pull an arm?
def MultiArmedBandit(bandits):
    # Create all the arms for simulation
    sim = Simulation(bandits=bandits, mean_low=20, mean_high=100,
                     stdev_low=3, stdev_high=10)

    # Plot 10 of the arms for illustrative purposes
    sim.plotArms(range(0,bandits,int(bandits/10)))

    # For reference, just randomly choose arms
    sim.plotRand()

    # The epsilon-greedy algorithm
    sim.plotEpsGreedy()

    # Bayesian Bandit
    sim.plotBayes()

MultiArmedBandit(bandits=10)
