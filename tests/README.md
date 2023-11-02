********************
Description of tests:

1-potential (updated):
--> runs 8 kmc steps, updating only the charge and potential after each step
--> the current is not computed, and the temperature is fixed to 300K

2-globaltemp (NOT updated):
--> runs X kmc steps, updating the charge, potential, and dissipated power after each step
--> uses a global temperature approximation to update a single device temperature using the dissipated power

3-localtemp (NOT updated):
--> runs X kmc steps, updating the charge, potential, dissipated power, and site-resolved heat distribution 
--> tests all modules of the code

*****************
How to run test-1:

--> cd ./1-potential/
--> ../../bin/runKMC ./parameters.txt

How to check test-1:
--> diff ./Results_20.000000/snapshot_8.xyz ./compare/Results_20.000000/snapshot_8.xyz

Notes: 
--> check_tests.py is missing some edge cases, do not use.
--> all tests have a fixed random seed so they should be exactly reproducible
