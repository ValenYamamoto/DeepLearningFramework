#include <iostream>
#include "../Tensor.hpp"
#include "Layer.hpp"
#include "Sequential.hpp"

Sequential::Sequential( int numLayers, Layer** layers ) : numLayers{numLayers}, layers{new Layer*[numLayers]}, intermediates{new Tensor[ numLayers ]} {
	int i;
	for( i=0; i<numLayers; i++ ) {
		this->layers[i] = layers[i];
	}
	parameters = new Tensor*[numLayers];
	for( i=0; i<numLayers; i++ ) {
		this->parameters[i] = layers[i]->getParameters()[0];
	}
}

Sequential::~Sequential() {
	delete[] layers;
	delete[] intermediates;
	delete[] parameters;
}

void Sequential::add( Layer *newLayer ) {
	numLayers++;
	Layer** newLayers = new Layer*[numLayers];
	int i;
	for( i=0; i<numLayers-1; i++ ) {
		newLayers[i] = layers[i];
	}
	delete[] layers;
	layers = newLayers;
}

Tensor Sequential::forward( Tensor& input ) {
	Tensor result = layers[0]->forward( input );
	if( numLayers == 1 ) {
		return result;
	}
	intermediates[0] = result;
	int i;
	for( i=0; i<numLayers-2; i++ ) {
		intermediates[i+1] = layers[i+1]->forward( intermediates[0] );
	}
	return layers[ numLayers-1 ]->forward( intermediates[ numLayers-2 ] );
}

