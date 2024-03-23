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
		std::vector<int> site_layer;													// layerID for each site
		//std::string method;															// simulation type: rejection or rejection free?
		double freq;                                                   	 				// attempt frequency of this set of events
		RandomNumberGenerator random_generator;											// random number generator object for this KMC process
		std::vector<int> affected_neighborhood;											// indices of the site-neighbor pairs

		KMCProcess(Device &device, double freq_); 										// constructor divides the device up into layers and initializes events
		void update_events_and_rates(Device &device, 									// updates all event_types and rates 
									 EVENTTYPE *event_type, 
									 double *event_prob); 
		void execute_event(Device &device, EVENTTYPE sel_event_type, int i, int j);		// executes event of type sel_event_type between i and j			
		void update_affected_neighborhood(int event_idx, Device &device);				// updates the affected_neighborhood of the KMC step
		std::map<std::string, double>  executeKMCStep(GPUBuffers gpubuf, Device &device, double *step_time);		// executes one step on the device and updates kmc_time with the event time
		
		
		// Physical Constants
		double kB = 8.617333262e-5;        // [eV/K]
		double q = 1.60217663e-19;         // [C]
		double h_bar_sq = 4.3957e-67;      // [(Js)^2]
		double h_bar = 1.054571817e-34;    // [Js]
		double h_bar_eV = 6.582119569e-16; // [eV]
		double m_0 = 9.11e-31;             // [kg]
		double eV_to_J = 1.6e-19;          // [C]
	
}; 
