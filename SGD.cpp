#include <iostream>
#include "Tensor.hpp"
#include "SGD.hpp"

SGD::SGD( int numParam, Tensor** parameters, double alpha ) : numParam{numParam}, parameters{new Tensor*[numParam]}, alpha{alpha} {
	int i;
	for( i=0; i<numParam; i++ ) {
		this->parameters[i] = parameters[i];
	}
}

SGD::~SGD() {
	delete[] parameters;
}

void SGD::zero() {
	int i;
	for( i=0; i<numParam; i++ ) {
		parameters[i]->clearGrad();
	}
}

void SGD::step( bool zero ) {
	int i;
	for( i=0; i<numParam; i++ ) {
		parameters[i]->update( parameters[i]->getGrad() * alpha );
		if( zero ) {
			parameters[i]->clearGrad();
		}
	}
}
