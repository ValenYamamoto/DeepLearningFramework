#include <tuple>
#include <iostream>
#include "LinearWithFunction.hpp"
#include "Layer.hpp"
#include "../Tensor.hpp"

LinearWithFunction::LinearWithFunction( int numInputs, int numOutputs, ActivationFunction activation, ActivationFunction backprop ) : numInputs{numInputs}, numOutputs{numOutputs} {
	parameters = new Tensor*[1];
	weight = Tensor::random( numInputs, numOutputs, true );
	parameters[0] = &weight;
	this->activation = activation;
	this->backprop = backprop;
}

LinearWithFunction::~LinearWithFunction() {
	delete[] parameters;
}

Tensor LinearWithFunction::forward( Tensor& input ) {
	Tensor result = input.mm( weight );
	result.applyActivationFunction( activation );
	result.setBackpropFunction( backprop );
	return input.mm( weight );
}
