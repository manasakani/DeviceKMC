# DeviceKMC
KMC simulator for VCM RRAM, refactored for EuroHack2023.

Models the time evolution of conductive filaments in atomically-resolved filamentary RRAM devices under varying potential and thermal gradients. Currently set up for a TiN-HfO2/Ti-TiN stack. 

To compile and run: $ make; ./bin/runKMC parameters.txt

Code structure: the main loop is in src/kmc_main.cpp -> each loop (1) recalculates the device fields, (2) executes a kinetic Monte Carlo perturbation of the vacancy distribution based on the residence time algorithm, and (3) increments the simulation time.
