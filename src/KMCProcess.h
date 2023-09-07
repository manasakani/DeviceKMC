// KMC Process class
#ifndef KMC_H
#define KMC_H
#include "Device.h"
#include "random_num.h"
#include <list>

// struct for a single KMC event
/*Key for event_type:
0 - Vacancy/Ion Pair Generation
1 - Vacancy/Ion Pair Recombination
2 - Vacancy Diffusion
3 - Ion Diffusion*/
struct Event{
	int ind1, ind2; //index in site list of site1 and site2 participating in this event
    int event_type; // type of event (see key above)    
    double prob; //probability of event
    Event* next;

    Event()
    {
        ind1 = 0;
        ind2 = 0;
        event_type = 0;
        prob = 0.0;
        next = NULL;
    }

    Event(int ind1_, int ind2_, int event_type_, double prob_)
    {
        this->ind1 = ind1_;
        this->ind2 = ind2_;
        this->event_type = event_type_;
        this->prob = prob_;   
        this->next = NULL;
    }
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
		KMCProcess(Device* device);
		
		// executes one step on the device and returns the time taken for it
		double executeKMCStep(Device* device, double freq, std::vector<double> lattice, bool pbc);
		
	private:
		RandomNumberGenerator random_generator;							// random number generator object for this KMC process
		std::vector<double> site_layer;									// layerID for each site
		
		// Physical Constants
		double kB = 8.617333262e-5;        // [eV/K]
		double q = 1.60217663e-19;         // [C]
		double h_bar_sq = 4.3957e-67;      // [(Js)^2]
		double h_bar = 1.054571817e-34;    // [Js]
		double h_bar_eV = 6.582119569e-16; // [eV]
		double m_0 = 9.11e-31;             // [kg]
		double eV_to_J = 1.6e-19;          // [C]
		
		Event* pick_and_get_event(std::list<Event>& event_list, int event_list_size, double Psum);
		
		void execute_event(Site* site_1, Site* site_2, int &event_type, int &charge_1, int &charge_2);
		
		/*void buildEventList(Device device);
		stepRejection();
		stepRejectionFree();*/
	
};
#endif 
