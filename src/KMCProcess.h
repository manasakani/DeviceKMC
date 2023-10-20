// KMC Process
#pragma once
#include "Device.h"
#include "random_num.h"
#include "gpu_buffers.h"
#include "utils.h"
#include <algorithm>
#include <list>
#include <numeric>
#include <string>
#include <vector>

class KMCProcess{
				
	public:
	    std::vector<Layer> layers;
		std::vector<int> site_layer;										// layerID for each site
		//std::string method;												// simulation type: rejection or rejection free?
		double freq;                                                   	 	// attempt frequency of this set of events
		KMCProcess(Device* device, double freq_); 

		// executes one step on the device and returns the time taken for it
		double executeKMCStep(GPUBuffers gpubuf, Device &device);
		// double executeKMCStep_gpu(GPUBuffers gpubuf, Device &device);
		
	private:
		RandomNumberGenerator random_generator;							// random number generator object for this KMC process
		
		// Physical Constants
		double kB = 8.617333262e-5;        // [eV/K]
		double q = 1.60217663e-19;         // [C]
		double h_bar_sq = 4.3957e-67;      // [(Js)^2]
		double h_bar = 1.054571817e-34;    // [Js]
		double h_bar_eV = 6.582119569e-16; // [eV]
		double m_0 = 9.11e-31;             // [kg]
		double eV_to_J = 1.6e-19;          // [C]
	
}; 
