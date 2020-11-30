#include "Tensor.hpp"
#include <vector>
#include <string>
#include <iostream>
#define DEBUG 0

namespace {
	const Tensor* const * createCreators( const Tensor* const *creators, Tensor::CreationOp creationOp ) {
		switch( creationOp ) {
			case Tensor::ADD:
				std::cout << "case add " << std::endl;
				return new const Tensor* const[2]{ creators[0], creators[1] };
			default:
				return nullptr;	
		}
	}
}

Tensor::Tensor() {
	#if DEBUG
	std::cout << "No arg Constructor" << std::endl;
	#endif
	size = 0;
	data = nullptr;
	creators = nullptr;
	grad = nullptr;
	creationOp=NONE;
}

Tensor::Tensor( long unsigned int size, double *data, const Tensor* const creators[], CreationOp creationOp ) : size{size}, grad{nullptr}, data{new double[size] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp ) } {
	#if DEBUG
	std::cout << "Array Constructor" << std::endl;
	#endif
	for( unsigned int i=0; i<size; i++ ) {
		this->data[i] = data[i];
	}
}

Tensor::Tensor( std::vector<double> values, const Tensor* const creators[], CreationOp creationOp ) : size{values.size()}, grad{nullptr}, data{new double[size] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )} {
	#if DEBUG
	std::cout << "Vector Constructor" << std::endl;
	#endif
	std::vector<double>::iterator dataPointer = values.begin();
	for( unsigned int i=0; dataPointer<values.end(); i++, dataPointer++ ) {
		data[i] = *dataPointer;
	}
}

Tensor::Tensor( const Tensor& original ) : size{original.size}, data{new double[size]}, creationOp{original.creationOp}, creators{createCreators(original.creators, creationOp) } {
	#if DEBUG
	std::cout << "Copy Constructor" << std::endl;
	#endif
	for( unsigned i=0; i<size; i++ ) {
		data[i] = original.data[i];
	}
	if( original.grad == nullptr ) {
		grad = nullptr;
	} else {
		grad = new Tensor(*original.grad);
	}
}

Tensor::~Tensor() {
	if( data != nullptr ) {
		delete[] data;
	}
	if( creators != nullptr ) {
		delete[] creators;
	}
	if( grad != nullptr ) {
		delete grad;
	}
}

Tensor& Tensor::operator =( const Tensor &right ) {
	#if DEBUG
	std::cout << "= operator Constructor" << std::endl;
	#endif
	if( this != &right ) {
		if( size != right.size ) {
			if( data != nullptr ) {
				delete[] data;
			}
			data = new double[size];
		}
		if( creators != nullptr ) {
			delete[] creators;
		}
		if( right.grad != nullptr ) {
			if( grad == nullptr ) {
				grad = new Tensor();
			}
			*grad = *( right.grad );
		} else {
			if( grad != nullptr ) {
				delete grad;
				grad = nullptr;
			}
		}
		
		this->size = right.size;
		unsigned int i;
		for( i=0; i<size; i++ ) {
			this->grad[i] = right.grad[i];
		}
		creationOp = right.creationOp;
		createCreators( right.creators, creationOp );
	}
	return *this;
}

Tensor Tensor::operator +( const Tensor &right ) {
	double *addData = new double[size];

	unsigned int i;
	for( i=0; i<size; i++ ) {
		addData[i] = right.data[i] + this->data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result = Tensor( size, addData, c, ADD );

	delete[] addData;
	delete[] c;

	return result;
}

void Tensor::backward( Tensor grad ) const {
	if( this->grad != nullptr ) {
		std::cout << "HERE" << std::endl;
		this->grad = new Tensor( grad );
	} else {
		delete this->grad;
		this->grad = new Tensor( grad );
	}
	
	if( creationOp == ADD ) {
		creators[0]->backward( grad );
		creators[1]->backward( grad );
	}
}

Tensor Tensor::getGrad() {
	return *grad;
}

std::string Tensor::to_string() {
	std::string s;
	s += "<" + std::to_string( data[0] );
	for( unsigned int i=1; i<size; i++ ) {
		s += ", " + std::to_string( data[i] );
	}
	s += ">";
	return s;
}
