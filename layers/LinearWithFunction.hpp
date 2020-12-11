#ifndef LINEARBACKPROP_H
#define LINEARBACKPROP_H

#include <functional>
#include "../Tensor.hpp"
#include "Layer.hpp"

class LinearWithFunction : public Layer {
	using ActivationFunction = std::function<double(double)>;
	public:
		LinearWithFunction( int numInputs, int numOutputs, ActivationFunction activation, ActivationFunction backprop );

		~LinearWithFunction();

		Tensor forward( Tensor& input );

	private:
		int numInputs;
		int numOutputs;
		Tensor weight;
		ActivationFunction activation;
		ActivationFunction backprop;
		
};
#endif
