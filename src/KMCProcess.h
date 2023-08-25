// KMC Process class
#ifndef KMC_H
#define KMC_H
#include "Device.h"
#include "random_num.h"

// struct for a single KMC event
//struct Event{
//};

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
		//std::string method;												//simulation type: rejection or rejection free?
		KMCProcess(Device* device);
		
		// executes one step on the device and returns the time taken for it
		double executeKMCStep(Device device);
		
	private:
		RandomNumberGenerator random_generator;							// random number generator object for this KMC process
		std::vector<double> site_layer;									// layerID for each site
		
		/*void buildEventList(Device device);
		void pickEvent(std::vector<Event> event_list);
		executeEvent(Event selected_event);
		stepRejection();
		stepRejectionFree();*/
	
};
#endif 
