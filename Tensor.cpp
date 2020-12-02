#include "Tensor.hpp"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#define DEBUG 0

namespace {
	const Tensor* const * createCreators( const Tensor* const *creators, Tensor::CreationOp creationOp ) {
		switch( creationOp ) {
			case Tensor::ADD:
				return new const Tensor* const[2]{ creators[0], creators[1] };
			case Tensor::NEG:
				return new const Tensor* const[1]{ creators[0] };
			default:
				return nullptr;	
		}
	}
}

int Tensor::nextID = 1;
Tensor::Tensor() {
	#if DEBUG
	std::cout << "No arg Constructor" << std::endl;
	#endif
	size = 1;
	data = nullptr;
	creators = nullptr;
	grad = nullptr;
	creationOp=NONE;
	autograd = false;
}

Tensor::Tensor( long unsigned int size, double *data, bool autograd, const Tensor* const creators[], CreationOp creationOp, int id ) : size{size}, grad{nullptr}, data{new double[size] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Array Constructor" << std::endl;
	#endif
	for( unsigned int i=0; i<size; i++ ) {
		this->data[i] = data[i];
	}
	if( creators != nullptr ) {
		createChildren();
	}
}

Tensor::Tensor( std::vector<double> values, bool autograd, const Tensor* const creators[], CreationOp creationOp, int id ) : size{values.size()}, grad{nullptr}, data{new double[size] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Vector Constructor" << std::endl;
	#endif
	std::vector<double>::iterator dataPointer = values.begin();
	for( unsigned int i=0; dataPointer<values.end(); i++, dataPointer++ ) {
		data[i] = *dataPointer;
	}
	if( creators != nullptr ) {
		createChildren();
	}
}

Tensor::Tensor( const Tensor& original ) : size{original.size}, data{new double[size]}, creationOp{original.creationOp}, creators{createCreators(original.creators, creationOp) }, autograd{original.autograd}, id{original.id} {
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
			data = new double[right.size];
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
			this->data[i] = right.data[i];
		}
		creationOp = right.creationOp;
		creators = createCreators( right.creators, creationOp );
		autograd = right.autograd;
		id = right.id;
	}
	return *this;
}

Tensor Tensor::operator +( const Tensor &right ) {
	double *addData = new double[right.size];

	unsigned int i;
	for( i=0; i<size; i++ ) {
		addData[i] = right.data[i] + this->data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd ) {
		result = Tensor{ size, addData, true, c, ADD };
	} else {
		result = Tensor{ size, addData };
	}

	delete[] addData;
	delete[] c;

	return result;
}

Tensor Tensor::operator -() {
	Tensor result;
	double *negateData = new double[ size ];
	unsigned int i;
	for( i=0; i<size; i++ ) {
		negateData[i] = data[i] * -1;
	}
	if( autograd ) {
		const Tensor* const *c = new const Tensor* const[1]{ this };
		result = Tensor{ size, negateData, true, c, NEG };
		std::cout << "NEG " << children.size() << std::endl;
		delete[] c;
	} else {
		result = Tensor{ size, negateData };
	}

	delete[] negateData;
	
	return result;
}

void Tensor::backward( Tensor grad, const Tensor* gradOrigin ) const {
	if( autograd ) {
		if( gradOrigin != nullptr ) {
			if( children[ gradOrigin->id ] == 0 ) {
				throw 0;
			} else {
				children[ gradOrigin->id ] -= 1;
			}
		}
		if( this->grad == nullptr ) {
			this->grad = new Tensor( grad );
		} else {
			*( this->grad ) = *(this->grad ) + grad ;
		}
		if( creators != nullptr && ( gradFromAllChildren() || gradOrigin == nullptr ) ) {
			switch( creationOp ){
				case ADD:
					creators[0]->backward( grad, this );
					creators[1]->backward( grad, this );
					break;
				case NEG:
					creators[0]->backward( -grad );
					break;
				default:
					;
			}
			
		}
	}
}

Tensor Tensor::getGrad() {
	if( grad != nullptr ) {
		return *grad;
	}
	return Tensor();
}

std::string Tensor::to_string() const {
	if( data != nullptr ) {
		std::string s;
		s += "<" + std::to_string( data[0] );
		for( unsigned int i=1; i<size; i++ ) {
			s += ", " + std::to_string( data[i] );
		}
		s += ">";
		return s;
	}
	return "";
}

void Tensor::createChildren() const {
	switch( creationOp ) {
		case ADD:
			creators[0]->addChild( id );
			creators[1]->addChild( id );
			break;
		case NEG:
			creators[0]->addChild( id );
			break;
		default:
			;
	}
}

void Tensor::addChild( int id ) const {
	if( children.count( id ) == 0 ) {
		children[ id ] = 1;
	} else {
		children[id] += 1;
	}
}

int Tensor::createTensorId() {
	nextID++;
	return nextID-1;
}

bool Tensor::gradFromAllChildren() const {
	std::map<int, int>::iterator it = children.begin();
	for( ; it!=children.end(); it++ ) {
		if( it->second != 0 ) {
			std::cout << it->first << " " << it->second << std::endl;
			return false;
		}
	}
	return true;
}

bool Tensor::getAutograd() const {
	return autograd;
}
