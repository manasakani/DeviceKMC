//Device class
#include <string>
#include <vector>
#include <random>

#ifndef DEVICE_H
#define DEVICE_H

// Base struct for individual vacancy or atom:
struct Site{
	
    int ind; 															// index in neighbor list
    std::vector<double> pos;    										// xyz position
    std::string element;												// atomic element ('d' for defect)
    bool isdefect; 														// is this a interstitial position?

    Site();
	void init_site(int ind_, double x_, double y_, double z_, std::string element);
	void disp_site();

};

// generic graph
struct Graph{
      
    int N;																// number of nodes in the graph
    std::vector<std::vector<int>> l;									// list of arrays which define the neighbor connections
    
    Graph(){}
      
    void initialize(int N){ 
        this->N = N;
        l.resize(N);
    }   

    void addEdge(int x, int y){ 
        l[x].push_back(y);
        l[y].push_back(x);
    }   
    ~Graph(){}
};

// A device is a collection of sites seperated into layers, and a neighbor list
class Device{
	
	public:
	    int N = 0;														// number of sites in this device
        std::vector<Site> sites;										// list of sites in this device
        Graph site_neighbors;											// list of neighbors of each site (including defects)
        Graph atom_neighbors;											// list of neighbors of each atom (excluding defects)
        double nn_dist;													// neighbor distance
        
        //std::vector<double> site_layer;								// layerID for each site
        std::vector<double> site_charge;								// charge of each site
        std::vector<double> site_potential;								// potential of each site
        std::vector<double> site_power;                                 // power of each site
        std::vector<double> site_temperature;							// temperature of each site
        
        // constructor from input xyz file(s)
        Device(std::vector<std::string>& xyz_files, std::vector<double> lattice, bool shift, std::vector<double> shifts, bool pbc, double nn_dist, double T_bg);
        
        // get number of sites with this element
        int get_num_of_element(std::string element_);
        
        // remove a specific percentage of oxygen from the lattice (convert to vacancies)
        void makeSubstoichiometric(double vacancy_concentration);
        
        // update the neighbor list of just the atoms (excluding the defects)
        void updateAtomNeighborList();
        
        void updateCharge();
        void updatePotential();
        void updatePower();
	    void updateTemperature();
	    
	    void writeSnapshot(std::string filename, std::string foldername);

    private:
        
	    std::vector<double> lattice;									// size of device box
	    bool pbc;														// is device periodic in the lateral directions?
	    
	    // initialize site_neighbors depending on nn_dist	    
        void constructSiteNeighborList();
        //void initLaplacian();
	
};
#endif 
