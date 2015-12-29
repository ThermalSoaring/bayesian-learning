#
# Bayesian Learning Demo
#
# Try to learn the mean of a normal distribution with Bayes' theorem given a
# few different measurements (that aren't even normal). Just getting things
# to work properly on test data before using this this in the bandit problem.
#
# Note: interesting developing a way to make Sage numerically evaluate the
# inverse_erf function through using mpmath. Messy, but it works.
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

# Start out assuming we know sigma
var('t')
normalDist(t,x,sigma)=1/sqrt(2*pi*sigma^2)*e^(-1/2*(x-t)^2/sigma^2)
known_sigma=1/10
p(t,x)=normalDist(t,x,sigma=known_sigma)

pmin=4
pmax=6
measurements=[5, 5+5/10, 4+2/10, 6, 4]
colors=rainbow(len(measurements))
plots=[]

# Pull a lever and find reward for it numIter times
for i in range(0, len(measurements)):
    if i == 0:
        # Initially a uniform distribution
        initial_a=0
        initial_b=2
        uniformDist(t,a,b)=1/(b-a) # for t in [a,b]
        prior(t)=uniformDist(t,a=initial_a,b=initial_b)

        # Posterior is different for the first since we have limits of
        # integration due to the uniform distribution
        #eq(t)=prior(t)*p(t,x=measurements[i])
        #norm_factor=integral(eq,t,initial_a,initial_b)
        #print "norm_factor=",norm_factor
        #post(t)=eq(t)/norm_factor

        # Initially just use the measurement as the mean, using a uniform
        # distribution with limits of integration just results in the
        # normalization factor having erf() in it and evaluating to about zero
        # giving us divide by zero issues.
        post(t)=p(t,x=measurements[i])
    else:
        # Otherwise, our prior is the last iteration's posterior
        prior(t)=post(t)

        # Posterior
        eq(t)=prior(t)*p(t,x=measurements[i])
        norm_factor=integral(eq,t,-oo,oo)
        post(t)=eq(t)/norm_factor

    mean = solve(diff(post,t)==0,t)[0].right()
    #F(x) = integral(post(t),t,x,oo)
    #interval95 = solve(F(t)==95/100,t)[0].right()
    #interval95_r = eval_inverse_erf(interval95)
    #print N(interval95_r)

    # What is F(mean) ... duh, 1/2 since it's normal
    #Fmean = integral(post, t, mean, oo)
    #print "P(>mean) =", Fmean, "=", N(Fmean)

    plots.append(
        #plot(prior,t,xmin=pmin,xmax=pmax,
        #    legend_label='prior'+str(i),color=colors[2*i])+
        plot(post,t,xmin=pmin,xmax=pmax,
            legend_label='posterior'+str(i),color=colors[i],
            title='Probabilities',axes_labels=['$t$','$P(t)$']))

s=sum(plots)
s.set_legend_options(loc=1)
show(s)
