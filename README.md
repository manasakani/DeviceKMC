# DeviceKMC
KMC simulator for VCM RRAM, initially refactored for EuroHack2023.

Models the time evolution of conductive filaments in atomically-resolved filamentary RRAM devices under varying potential and thermal gradients. Currently set up to model a TiN-HfO2/Ti-TiN stack. 

Notes:
mpirun -np N -hostfile ./hosts ./bin/runKMC parameters.txt
