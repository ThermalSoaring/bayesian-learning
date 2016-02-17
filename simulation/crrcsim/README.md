crrcsim Simulation
------------------------
You can follow the instructions at the top of run `run_demo.sh` and then run
it. This will produce a 'simulation.csv' file which you can then discretize
with `python3 discretize.py`, which will output a
'simulation_discretized.csv' file. This can then be run through CaMML to
learn the Bayesian Network or Dynamic Bayesian Network. Finally, exporting
these learned networks as *.dne* files can be opened in SamIAm. However,
SamIAm will only rearrange files if you are working with a *.net* file, so
create a new network and copy everything from the *.dne* file to the .net file.
Then you can use SamIAm's auto arrange functions and save the positions of the
nodes.

You'll find the results in the *data/* folder.
