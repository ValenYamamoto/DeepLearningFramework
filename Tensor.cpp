#include "Tensor.hpp"
#include <vector>
#include <string>


Tensor::Tensor() {
	size = 0;
	data = nullptr;
	creators = nullptr;
	grad = nullptr;
	creationOp=NONE;
}

Tensor::Tensor( int size, double *data, Tensor *creators, CreationOp creationOp ) : size{size}, grad{nullptr}, data{new double[size] } {
	for( int i=0; i<size; i++ ) {
		this->data[i] = data[i];
	}

	createCreators( creators, creationOp );
}

template <typename Iter>
Tensor::Tensor( Iter values, Tensor *creators, CreationOp creationOp ) : size{values.size()}, grad{nullptr}, data{new double[size] } {
	typename Iter::iterator dataPointer = values.begin();
	for( int i=0; dataPointer<values.end(); i++, dataPointer++ ) {
		data[i] = *dataPointer;
	}
	createCreators( creators, creationOp );
}

Tensor::Tensor( const Tensor& original ) : size{original.size}, data{new double[size]} {
	int i;
	for( i=0; i<size; i++ ) {
		data[i] = original.data[i];
	}
	createCreators( original.creators, original.creationOp );
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
		int i;
		for( i=0; i<size; i++ ) {
			this->grad[i] = right.grad[i];
		}
		createCreators( right.creators, right.creationOp );
	}
	return *this;
}

Tensor Tensor::operator +( const Tensor &right ) {
	double *addData = new double[size];

	int i;
	for( i=0; i<size; i++ ) {
		addData[i] = right.data[i] + this->data[i];
	}

	Tensor c[2] = { *this, right };

	Tensor result = Tensor( size, addData, c, ADD );

	return result;
}

void Tensor::backward( Tensor grad ) {
	if( this->grad == nullptr ) {
		this->grad = new Tensor();
	}
	*(this->grad) = grad;
	
	if( creationOp == ADD ) {
		creators[0].backward( *(this->grad) );
		creators[1].backward( *(this->grad) );
	}
}

Tensor Tensor::getGrad() {
	return *grad;
}

std::string Tensor::to_string() {
	std::string s;
	s += "<" + std::to_string( data[0] );
	for( int i=1; i<size; i++ ) {
		s += ", " + std::to_string( data[i] );
	}
	s += ">";
	return s;
}

void Tensor::createCreators( Tensor *creators, CreationOp creationOp ) {
	this->creationOp = creationOp;
	switch( creationOp ) {
		case ADD:
			this->creators = new Tensor[2];
			this->creators[0] = creators[0];
			this->creators[1] = creators[1];
			break;
		default:
			this->creators = nullptr;
	}
}
