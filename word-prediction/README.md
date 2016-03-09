# Predicting Next Word with a Bayesian Network

Created this for Intro to AI class as a proof-of-concept of using a BN. This
project is very much related to the senior project, and through it I found
[Pomegranate](https://github.com/jmschrei/pomegranate), which is a Python
library that supports BNs. This demo is using discrete variables, but it
appears to support approximating complex continuous distributions with a
mixture of Gaussians.

The input is text files of books from [Project
Gutenburg](https://www.gutenberg.org/), which are used to learn the prior and
conditional probability tables for the network. The network is _n_ nodes, where
a given word is dependent on all previous words.

The idea is to type in some words and the program will predict the next word
taking as observations for the network the last _n-1_ words.

Example usage:

	$ python3 ./word_prediction.py -n 3 alices_adventures_in_wonderland.txt
	>>> we went
	 ('to', 1.0)
	>>> they walked
	 ('off', 1.0)
	>>> they walk
	 ('with', 0.25)
	 ('a', 0.25)
	 ('Coming', 0.25)
	 ('the', 0.25)
	>>> the
	 ('Queen', 0.11881813872649812)
	 ('Mock', 0.11097066413862176)
	 ('King', 0.10428187707378829)
	 ('Gryphon', 0.09738241955021937)
	 ('Hatter', 0.07599936798862457)
	 ('Duchess', 0.0573550323905835)
