#pragma once
#include <random>

// class RandomNumberGenerator
// {

// public:
// 	RandomNumberGenerator() : rng(0) {}

// 	void setSeed(unsigned int seed)
// 	{
// 		rng.seed(seed);
// 	}

// 	double getRandomNumber()
// 	{
// 		std::uniform_real_distribution<double> distribution(0.0, 1.0);
// 		return distribution(rng);
// 	}

// private:
// 	std::mt19937 rng;
// };


class RandomNumberGenerator
{

public:
	RandomNumberGenerator() : rng(0) 
    {}

	void setSeed(unsigned int seed)
	{
		rng.seed(seed);
        distribution = new  std::uniform_real_distribution<double>(0.0, 1.0);
	}

	double getRandomNumber()
	{
		// std::uniform_real_distribution<double> distribution(0.0, 1.0);
		return (*distribution)(rng);
	}

private:
	std::mt19937 rng;
    std::uniform_real_distribution<double> *distribution;
};
