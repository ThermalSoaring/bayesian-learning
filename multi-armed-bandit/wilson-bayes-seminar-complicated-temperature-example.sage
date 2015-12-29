#
# Example of updating belief about the temperature in the room based on two
# independent measurements with a prior belief that it is a normal distribution
# with a mean of 18 units and a standard deviation of 0.3.
#
# Basically the plot on page 32 of:
# https://www.cs.tcd.ie/disciplines/statistics/statica/statica_web/Wilson_Bayes_seminar.pdf
#

var('t')
p(t,x1,x2,sigma)=1/2/pi/sigma^2*e^(-1/2/sigma^2*((x1-t)^2+(x2-t)^2))
pT(t,x,sigma)=1/sqrt(2*pi*sigma^2)*e^(-1/2*(x-t)^2/sigma^2)

ex_x1=18+1/10
ex_x2=18+4/10
ex_sigma=3/10

s=\
    plot(p(t,x1=ex_x1,x2=ex_x1,sigma=ex_sigma),xmin=17,xmax=19,\
        legend_label='x1=x2='+str(N(ex_x1,12)),color='#FF0000',\
        title='Likelihood',axes_labels=['$t$','$p(t)$'])+\
    plot(p(t,x1=ex_x1,x2=ex_x2,sigma=ex_sigma),xmin=17,xmax=19,\
        legend_label='x1='+str(N(ex_x1,12))+', x2='+str(N(ex_x2,12)),color='#00FF00')
s.set_legend_options(loc=1)
#show(s)

prior(t)=pT(t,x=18,sigma=ex_sigma)
eq1(t)=prior(t)*p(t,x1=ex_x1,x2=ex_x1,sigma=ex_sigma)
# Note: this integral comes out to be zero if you use floating point numbers,
# so you must make sure all of your numbers are fractions
norm_factor1=integral(eq1,t,-oo,oo)
eq2(t)=prior(t)*p(t,x1=ex_x1,x2=ex_x2,sigma=ex_sigma)
norm_factor2=integral(eq2,t,-oo,oo)
post1(t)=eq1(t)/norm_factor1
post2(t)=eq2(t)/norm_factor2

s=\
    plot(prior(t),t,xmin=17,xmax=19,
        legend_label='$prior(t)$',color='#000000')+\
    plot(post1(t),t,xmin=17,xmax=19,\
        legend_label='x1=x2='+str(N(ex_x1,12)),color='#FF0000',\
        title='Posterior',axes_labels=['$t$','$p_T(t|x_1,x_2)$'])+\
    plot(post2(t),t,xmin=17,xmax=19,\
        legend_label='x1='+str(N(ex_x1,12))+', x2='+str(N(ex_x2,12)),color='#00FF00')
s.set_legend_options(loc=1)
show(s)
