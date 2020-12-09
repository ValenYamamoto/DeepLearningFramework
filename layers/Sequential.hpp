#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../Tensor.hpp"
#include "Layer.hpp"

class Sequential: public Layer {
	public:
		Sequential( int numLayers, Layer** layers );

		~Sequential();

		void add( Layer *newLayer );

		Tensor forward( Tensor& input );

	private:
		int numLayers;
		Layer** layers;
		Tensor* intermediates;
};

#endif
