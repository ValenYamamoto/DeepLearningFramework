#ifndef LINEAR_H
#define LINEAR_H

#include "../Tensor.hpp"
#include "Layer.hpp"

class Linear : public Layer {
	public:
		Linear( int numInputs, int numOutputs );

		~Linear();

		Tensor forward( Tensor& input );

	private:
		int numInputs;
		int numOutputs;
		Tensor weight;
};
#endif
