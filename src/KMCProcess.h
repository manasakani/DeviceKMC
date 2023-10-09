// KMC Process
#pragma once
#include "Device.h"
#include "random_num.h"
#include "gpu_buffers.h"
#include <algorithm>
#include <list>
#include <numeric>
#include <string>
#include <vector>

// KMC events
enum EVENTTYPE
{
    VACANCY_GENERATION,
    VACANCY_RECOMBINATION,
    VACANCY_DIFFUSION,
    ION_DIFFUSION,
    NULL_EVENT
};


//Creates a device 'layer', with activation energies and types
struct Layer{
    std::string type;
    double E_gen_0, E_rec_1, E_diff_2, E_diff_3;
    double start_x, end_x;
    double init_vac_percentage; 
    Layer(){}
   
    void init_layer(std::string type_, double E_gen_0_, double E_rec_1_, double E_diff_2_, double E_diff_3_, double start_x_, double end_x_);
    void disp_layer();
};


class KMCProcess{
				
	public:
	    std::vector<Layer> layers;
		//std::string method;												//simulation type: rejection or rejection free?
		KMCProcess(Device* device, double freq_); 

		// executes one step on the device and returns the time taken for it
		double executeKMCStep(Device &device);
		double executeKMCStep_gpu(GPUBuffers gpubuf);
		
	private:
		RandomNumberGenerator random_generator;							// random number generator object for this KMC process
		std::vector<double> site_layer;									// layerID for each site
        double freq;                                                    // attempt frequency of this set of events
		
		// Physical Constants
		double kB = 8.617333262e-5;        // [eV/K]
		double q = 1.60217663e-19;         // [C]
		double h_bar_sq = 4.3957e-67;      // [(Js)^2]
		double h_bar = 1.054571817e-34;    // [Js]
		double h_bar_eV = 6.582119569e-16; // [eV]
		double m_0 = 9.11e-31;             // [kg]
		double eV_to_J = 1.6e-19;          // [C]
	
}; 
