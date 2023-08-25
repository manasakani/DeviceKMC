//***********************************
//  Global Input Structure Parameters
//***********************************

#include <vector>
#include <string>

#ifndef STRUCTURE_H
#define STRUCTURE_H

unsigned int rnd_seed_kmc = 1;

int numlayers = 5;

std::string layer_0_type = "contact";
double layer_0_E_gen_0 = 0.0;
double layer_0_E_rec_1 = 0.0;
double layer_0_E_diff_2 = 0.0;
double layer_0_E_diff_3 = 0.76;
double layer_0_start_x = -22.0;
double layer_0_end_x = 0.0;

std::string layer_1_type = "interface";
double layer_1_E_gen_0 = 3.93;
double layer_1_E_rec_1 = 0.0;
double layer_1_E_diff_2 = 1.09;
double layer_1_E_diff_3 = 0.76;
double layer_1_start_x = 0.0;
double layer_1_end_x = 3.0;

std::string layer_2_type = "oxide";
double layer_2_E_gen_0 = 3.93;
double layer_2_E_rec_1 = 0.0;
double layer_2_E_diff_2 = 1.09;
double layer_2_E_diff_3 = 0.76;
double layer_2_start_x = 3.0;
double layer_2_end_x = 49.3309;

std::string layer_3_type = "interface";
double layer_3_E_gen_0 = 1.66;
double layer_3_E_rec_1 = 0.0;
double layer_3_E_diff_2 = 1.09;
double layer_3_E_diff_3 = 0.76;
double layer_3_start_x = 49.3309;
double layer_3_end_x = 54.33894;

std::string layer_4_type = "contact";
double layer_4_E_gen_0 = 0.0;
double layer_4_E_rec_1 = 0.0;
double layer_4_E_diff_2 = 0.0;
double layer_4_E_diff_3 = 2.00; // CHANGED
double layer_4_start_x = 54.33894;
double layer_4_end_x = 90.0;

#endif
