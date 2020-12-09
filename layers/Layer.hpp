#ifndef LAYER_H
#define LAYER_H

#include "../Tensor.hpp"
class Layer {
	public:
		Tensor** getParameters() {
			return this->parameters;
		}

		virtual Tensor forward( Tensor& input ) = 0;
		Tensor** parameters;
};
#endif
