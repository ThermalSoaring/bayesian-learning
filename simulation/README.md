Learning on a Simulation
------------------------
This will learn where thermals are using data from a simulator. Presently
obtaining data from FlightGear in Python is too slow, so crrcsim is more
useful, using ArduPilot for an autopilot.

The _cpp-to-python_ code here is an example of how to have the Python code
receive autopilot data from the C++ program and then send back commands to the
C++ program for what to do. This is what will be used when working with the
actual Piccolo simulator.
