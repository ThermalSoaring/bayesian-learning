# Gaussian Process Regression

## GPR Thermal Soaring
Use Gaussian process regression to see how it learns the thermal field
and how the uncertainty changes with the number of observations taken at
random points.

## GPR vs. Bayesian
Try to visualize a comparison between GPR and the Bayesian parameter
estimation from the *gaussian-thermals/* directory. Since we can't
directly visualize a comparison with probability distributions of
parameters, we'll look at the most probable thermal field from MAP
estimates of the parameters. Now we use more realistic observation
points, along a path rather than random points over some area.

## GPR Varying Parameters
What happens when we change the GPR parameters? What hyperparameters are we
using? How does this compare with what Lawrance got in his thesis?

## GPR Simulation Data - Picking Nugget
Based on the simulation data, pick a reasonable nugget value for the GPR.

## GPR vs. Bayesian using Simulation Data
Use data from the simulator rather than data from a simple Gaussian-shaped
thermal.
