// KMC Process class
#include "random_num.h"
#include "KMCProcess.h"
#include "structure_input.h"
#include "Device.h"
#include "utils.h"
#include <iostream>


void Layer::init_layer(std::string type_, double E_gen_0_, double E_rec_1_, double E_diff_2_, double E_diff_3_, double start_x_, double end_x_){
    type = type_;
    E_gen_0 = E_gen_0_;
    E_rec_1 = E_rec_1_;
    E_diff_2 = E_diff_2_;
    E_diff_3 = E_diff_3_;
    start_x = start_x_;
    end_x = end_x_;
    init_vac_percentage = 0.0;
}

void Layer::disp_layer(){
    print("Layer of type " << type << " from " << start_x << " to " << end_x); 
}

KMCProcess::KMCProcess(Device* device){
	
	// initialize random number generator
	random_generator.setSeed(rnd_seed_kmc);
		
	//intialize device layers
	int layerID;
	Layer *layers = new Layer[numlayers];
    layers[0].init_layer(layer_0_type, layer_0_E_gen_0, layer_0_E_rec_1, layer_0_E_diff_2, layer_0_E_diff_3, layer_0_start_x, layer_0_end_x);
    layers[1].init_layer(layer_1_type, layer_1_E_gen_0, layer_1_E_rec_1, layer_1_E_diff_2, layer_1_E_diff_3, layer_1_start_x, layer_1_end_x);
    layers[2].init_layer(layer_2_type, layer_2_E_gen_0, layer_2_E_rec_1, layer_2_E_diff_2, layer_2_E_diff_3, layer_2_start_x, layer_2_end_x);
    layers[3].init_layer(layer_3_type, layer_3_E_gen_0, layer_3_E_rec_1, layer_3_E_diff_2, layer_3_E_diff_3, layer_3_start_x, layer_3_end_x);
    layers[4].init_layer(layer_4_type, layer_4_E_gen_0, layer_4_E_rec_1, layer_4_E_diff_2, layer_4_E_diff_3, layer_4_start_x, layer_4_end_x);
	
	//assign layersIDs for all the sites
	for(int i = 0; i < device->N; i++){
        layerID = 100;
        for(int j = 0; j < numlayers; j++){
            if (layers[j].start_x <= device->sites[i].pos[0] && device->sites[i].pos[0] <= layers[j].end_x){
                layerID = j;
            }
        }
        if (layerID == 100){
            print("Site #" << i << " is not inside the device!");
            abort();
        }
		site_layer.push_back(layerID);
    }
	
}

double executeKMCStep(Device device){
	
	// build event list
	
	// while loop
	    // select and execute event
	    // cleanup event list
}
