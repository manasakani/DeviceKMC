#include "input_parser.h"

KMCParameters::KMCParameters(std::string param_file){
	
    std::string line;
    std::ifstream input_file (param_file);
    
    if (input_file.is_open())
    {
      while ( std::getline (input_file,line) )
      {

        // Comment line
		if (line.substr(0, 2) == "//") {
            continue;
        } 
        line = trimAfterDoubleSlash(line);
        
        // random number seed
        if (line.find("rnd_seed ") != std::string::npos) {
			rnd_seed = read_int(line);
		}
        
        // option for restart calculation
        if (line.find("restart ") != std::string::npos) {
			restart = read_bool(line);
		}
		
		if (line.find("restart_xyz_file ") != std::string::npos) {
			restart_xyz_file = read_string(line);
		}
		
		//I/O
		if (line.find("log_freq ") != std::string::npos) {
			log_freq = read_int(line);
		}
		
		if (line.find("output_freq ") != std::string::npos) {
			output_freq = read_int(line);
		}
		
		if (line.find("log_to_file ") != std::string::npos) {
			log_to_file = read_bool(line);
		}
		
		if (line.find("verbose ") != std::string::npos) {
			log_to_file = read_bool(line);
		}
		
		// device atomic structure
		if (line.find("atom_xyz_file ") != std::string::npos) {
			atom_xyz_file = read_string(line);
		}
		
		if (line.find("interstitial_xyz_file ") != std::string::npos) {
			interstitial_xyz_file = read_string(line);
		}
		
		if (line.find("pristine ") != std::string::npos) {
			pristine = read_bool(line);
		}
		
		if (line.find("shift ") != std::string::npos) {
			shift = read_bool(line);
		}
		
		if (line.find("pbc ") != std::string::npos) {
			pbc = read_bool(line);
		}
		
		if (line.find("num_atoms_first_layer ") != std::string::npos) {
			num_atoms_first_layer = read_int(line);
		}

		if (line.find("num_layers_contact ") != std::string::npos) {
			num_layers_contact = read_int(line);
		}
		
		if (line.find("num_atoms_contact ") != std::string::npos) {
			num_atoms_contact = read_int(line);
		}

		if (line.find("num_atoms_reservoir ") != std::string::npos) {
			num_atoms_reservoir = read_int(line);
		}
		
		if (line.find("initial_vacancy_concentration ") != std::string::npos) {
			initial_vacancy_concentration = read_double(line);
		}
		
		if (line.find("nn_dist ") != std::string::npos) {
			nn_dist = read_double(line);
		}
		
		if (line.find("attempt_frequency ") != std::string::npos) {
			freq = read_double(line);
		}
		
		if (line.find("shifts ") != std::string::npos) {
			shifts = read_vec_double(line);
		}
		
		if (line.find("lattice ") != std::string::npos) {
			lattice = read_vec_double(line);
		}
		
		if (line.find("metals ") != std::string::npos) {
			std::vector<std::string>
			 metalsChar = read_vec_string(line);
            for (auto e: metalsChar){

				metals.push_back(update_element(e));
			}
		}
		
		// field solvers
		if (line.find("solve_potential ") != std::string::npos) {
			solve_potential = read_bool(line);
		}
		
		if (line.find("solve_current ") != std::string::npos) {
			solve_current = read_bool(line);
		}
		
		if (line.find("solve_heating_global ") != std::string::npos) {
			solve_heating_global = read_bool(line);
		}
		
		if (line.find("solve_heating_local ") != std::string::npos) {
			solve_heating_local = read_bool(line);
		}

		// Whether to execute events
		if (line.find("perturb_structure ") != std::string::npos) {
			perturb_structure = read_bool(line);
		}
		
		// Biasing scheme
		if (line.find("V_switch ") != std::string::npos) {
			V_switch = read_vec_double(line);
		}
		
		if (line.find("t_switch ") != std::string::npos) {
			t_switch = read_vec_double(line);
		}
		
		if (line.find("Icc ") != std::string::npos) {
			Icc = read_double(line);
		}

		if (line.find("Rs ") != std::string::npos)
		{
			Rs = read_double(line);
		}

		// for potential solver
		if (line.find("sigma ") != std::string::npos) {
			sigma = read_double(line);
		}
		
		if (line.find("epsilon ") != std::string::npos) {
			epsilon = read_double(line);
		}
		
		// for current solver (tunneling parameters)
		if (line.find("m_r ") != std::string::npos) {
			m_r = read_double(line);
		}
		
		if (line.find("V0 ") != std::string::npos) {
			V0 = read_double(line);
		}
		
		if (line.find("alpha ") != std::string::npos) {
			alpha = read_vec_double(line);
		}
		
		// for temperature solver
		if (line.find("k_therm ") != std::string::npos) {
			k_therm = read_double(line);
		}
		
		if (line.find("background_temp ") != std::string::npos) {
			background_temp = read_double(line);
		}
		
		if (line.find("dissipation_constant ") != std::string::npos) {
			dissipation_constant = read_double(line);
		}
		
		if (line.find("small_step ") != std::string::npos) {
			small_step = read_double(line);
		}
		
		if (line.find("event_time ") != std::string::npos) {
			event_time = read_double(line);
		}
		
		if (line.find("delta_t ") != std::string::npos) {
			delta_t = read_double(line);
		}
		
		if (line.find("delta ") != std::string::npos) {
			delta = read_double(line);
		}
		
		if (line.find("power_adjustment_term ") != std::string::npos) {
			power_adjustment_term = read_double(line);
		}
		
		if (line.find("L_char ") != std::string::npos) {
			L_char = read_double(line);
		}
		
		if (line.find("k_th_metal ") != std::string::npos) {
			k_th_metal = read_double(line);
		}
		
		if (line.find("k_th_non_vacancy ") != std::string::npos) {
			k_th_non_vacancy = read_double(line);
		}
		
		if (line.find("k_th_vacancies ") != std::string::npos) {
			k_th_vacancies = read_double(line);
		}
		
		if (line.find("c_p ") != std::string::npos) {
			c_p = read_double(line);
		}
		
		if (line.find("t_ox ") != std::string::npos) {
			t_ox = read_double(line);
		}
		
		if (line.find("A ") != std::string::npos) {
			std::vector<double> A_dims = read_vec_double(line);
			A = 1;
			for (auto d : A_dims){
				A *= d;
			}
		}
		
      }
      input_file.close();
    } 
    
    // initialize the expressions:
    set_expression_parameters();
}

std::string KMCParameters::trimAfterDoubleSlash(std::string& input) {
    size_t pos = input.find("//");
    if (pos != std::string::npos) {
        return input.substr(0, pos);
    } else {
        return input;
    }
}


bool KMCParameters::read_bool(std::string line){
	
	std::string lowerInput = line;
    
    if (lowerInput.find("1") != std::string::npos) {
        return true;
    } else if (lowerInput.find("0") != std::string::npos) {
        return false;
    } else {
        throw std::invalid_argument("Invalid input to read_bool: " + line);
    }
    	
}


unsigned int KMCParameters::read_unsigned_int(std::string line){
	
	std::istringstream stream(line);
    unsigned int value;
    
    if (!(stream >> value)) {
        throw std::invalid_argument("Invalid input: " + line);
    }

    return value;
}


int KMCParameters::read_int(std::string line){
	
	std::istringstream stream(line);
    std::string token;

    while (stream >> token) {
		if (token == "=") {
			if (stream >> token) {
				int value;
				if (std::istringstream(token) >> value) {
					return value;
				} else {
						throw std::invalid_argument("Invalid integer after equal sign: " + token);
				}
			}
		}
	}

	throw std::invalid_argument("Equal sign and integer not found in input: " + line);
    
}

double KMCParameters::read_double(std::string line){
	
	std::istringstream stream(line);
    double value = 0.0; 
    
    std::string token;
    while (stream >> token) {
        double tempValue;
        if (std::istringstream(token) >> tempValue) {
            value = tempValue;
        } else if (token.find_first_not_of("0123456789.eE-") == std::string::npos) {
            try {
                value = std::stod(token);
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument("Invalid double value in scientific notation: " + token);
            }
        }
    }
    
    if (value != 0.0) {
    return value;
	} else {
		throw std::invalid_argument("No double value found in input: " + line);
	}

}

std::string KMCParameters::read_string(std::string line){
	
	std::istringstream stream(line);
    std::string lastWord;
    std::string word;
    
    while (stream >> word) {
		lastWord = word;
	}

	return lastWord;
}

std::vector<double> KMCParameters::read_vec_double(std::string line){
	
	std::vector<double> doubleValues;
    std::istringstream stream(line);

    std::string token;
    while (stream >> token) {
        double value;
        if (std::istringstream(token) >> value) {
            doubleValues.push_back(value);
        }
    }

    return doubleValues;
	
}


std::vector<std::string> KMCParameters::read_vec_string(std::string line){
	
	std::vector<std::string> stringValues;
    std::istringstream stream(line);

    std::string token;
    bool foundEqualSign = false;

    while (stream >> token) {
        if (foundEqualSign) {
            stringValues.push_back(token);
        }
        if (token == "=") {
            foundEqualSign = true;
        }
    }

    return stringValues;
	
}


void KMCParameters::set_expression_parameters(){
	high_G = G_coeff * 1;
	low_G = G_coeff * 1e-8; 																						// [S]
	k = 8.987552e9 / epsilon;   																				  	// [N m^2 / C^2]
    k_th_interface = k_th_non_vacancy + (k_th_vacancies - k_th_non_vacancy) * initial_vacancy_concentration; 		// [W/mK]
    tau = k_th_interface / (L_char * L_char * c_p * 1e6);                                                   	 	// Thermal rate constant [1/s]
    m_e = m_r * m_0;            																				  	// [kg] 
}

void KMCParameters::print_to_file(){}