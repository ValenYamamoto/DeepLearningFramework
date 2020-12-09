#include "Linear.hpp"
#include "Layer.hpp"
#include "../Tensor.hpp"

Linear::Linear( int numInputs, int numOutputs ) : numInputs{numInputs}, numOutputs{numOutputs} {
	parameters = new Tensor*[1];
	weight = Tensor::random( numInputs, numOutputs, true );
	parameters[0] = &weight;
}

Linear::~Linear() {
	delete[] parameters;
}

Tensor Linear::forward( Tensor& input ) {
	return input.mm( weight );
}
