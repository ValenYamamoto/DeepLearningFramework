#include "Tensor.hpp"
#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <iostream>
#include <cstdlib>
#define DEBUG 0

namespace {
	const Tensor* const * createCreators( const Tensor* const *creators, Tensor::CreationOp creationOp ) {
		switch( creationOp ) {
			case Tensor::ADD:
			case Tensor::SUB:
			case Tensor::MUL:
			case Tensor::MM:
				return new const Tensor* const[2]{ creators[0], creators[1] };
			case Tensor::NEG:
			case Tensor::TRANSPOSE:
				return new const Tensor* const[1]{ creators[0] };
			default:
				return nullptr;	
		}
	}

	void naiveMatrixMult( int m, int n, int o, double *m1, double *m2, double *result ) {
		int i, j, k;
		double sum;
		for( i=0; i<m; i++ ) {
			for( j=0; j<o; j++ ) {
				sum = 0;
				for( k=0; k<n; k++ ) {
					sum += *( m1+m*i+k ) * *( m2+n*k+j );
				}
				*( result+m*i+j ) = sum;
			}
		}
	}

	void transposeMatrix( int r, int c, double *m, double *result ) {
		int i, j;
		for( i=0; i<r; i++ ) {
			for( j=0; j<c; j++ ) {
				*( result+j*r+i ) = *( m+c*i+j );
			}
		}
	}
	int totalElements( std::tuple<int, int> size ) {
		return std::get<0>( size ) * std::get<1>( size );
	}
}

int Tensor::nextID = 1;

Tensor::Tensor() {
	#if DEBUG
	std::cout << "No arg Constructor" << std::endl;
	#endif
	size = std::tuple<int, int>{ 1, 1 };
	data = nullptr;
	creators = nullptr;
	grad = nullptr;
	creationOp=NONE;
	autograd = false;
}

Tensor::Tensor( std::tuple<int, int> size, double *data, bool autograd, const Tensor* const creators[], CreationOp creationOp, int id ) : size{size}, grad{nullptr}, data{new double[totalElements(size)] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Array Constructor" << std::endl;
	#endif
	for( int i=0; i<totalElements(size); i++ ) {
		this->data[i] = data[i];
	}
	if( creators != nullptr ) {
		createChildren();
	}
}

Tensor::Tensor( std::vector<double> values, bool autograd, const Tensor* const creators[], CreationOp creationOp, int id ) : size{std::tuple<int, int>(values.size(), 1 )}, grad{nullptr}, data{new double[totalElements(size)] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Vector Constructor" << std::endl;
	#endif
	std::vector<double>::iterator dataPointer = values.begin();
	for( int i=0; dataPointer<values.end(); i++, dataPointer++ ) {
		data[i] = *dataPointer;
	}
	if( creators != nullptr ) {
		createChildren();
	}
}

Tensor::Tensor( const Tensor& original ) : size{original.size}, data{new double[totalElements(size)]}, creationOp{original.creationOp}, creators{createCreators(original.creators, creationOp) }, autograd{original.autograd}, id{original.id} {
	#if DEBUG
	std::cout << "Copy Constructor" << std::endl;
	#endif
	for( int i=0; i<totalElements(size); i++ ) {
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

Tensor Tensor::random( int rows, int cols ) {
	std::tuple<int, int> size = std::tuple<int, int>{ rows, cols };
	double *data = new double[ rows * cols ];
	int i;
	for( i=0; i<rows*cols; i++ ) {
		data[i] = 2 * ( std::rand() / (double) RAND_MAX ) - 1;
	}
	return Tensor{ size, data };
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
			data = new double[totalElements(right.size)];
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
		for( i=0; i<totalElements(size); i++ ) {
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
	double *addData = new double[totalElements(right.size)];

	int i;
	for( i=0; i<totalElements( size ); i++ ) {
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
	double *negateData = new double[ totalElements( size ) ];
	int i;
	for( i=0; i<totalElements( size ); i++ ) {
		negateData[i] = data[i] * -1;
	}
	if( autograd ) {
		const Tensor* const *c = new const Tensor* const[1]{ this };
		result = Tensor{ size, negateData, true, c, NEG };
		delete[] c;
	} else {
		result = Tensor{ size, negateData };
	}

	delete[] negateData;
	
	return result;
}

Tensor Tensor::operator -( const Tensor &right ) {
	double *addData = new double[totalElements(right.size)];

	int i;
	for( i=0; i<totalElements( size ); i++ ) {
		addData[i] = this->data[i] - right.data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd ) {
		result = Tensor{ size, addData, true, c, SUB };
	} else {
		result = Tensor{ size, addData };
	}

	delete[] addData;
	delete[] c;

	return result;
}

Tensor Tensor::operator *( const Tensor &right ) {
	double *mulData = new double[totalElements(right.size)];

	int i;
	for( i=0; i<totalElements(size); i++ ) {
		mulData[i] = this->data[i] * right.data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd ) {
		result = Tensor{ size, mulData, true, c, MUL };
	} else {
		result = Tensor{ size, mulData };
	}

	delete[] mulData;
	delete[] c;

	return result;
}

Tensor Tensor::mm( const Tensor &right ) {
	double *mmData = new double[ std::get<0>( size ) * std::get<1>( right.size ) ];

	naiveMatrixMult( std::get<0>( size ), std::get<1>( size ), std::get<1>( right.size ), data, right.data, mmData );

	Tensor result;
	if( autograd && right.autograd ) {
		const Tensor* const *c = new const Tensor* const[2]{ this, &right };
		result = Tensor{ std::tuple<int, int>{ std::get<0>( size ), std::get<1>( right.size )}, mmData, true, c, MM }; 
		delete[] c;
	} else {
		result = Tensor{ std::tuple<int, int>{ std::get<0>( size ), std::get<1>( right.size )}, mmData };
	}

	delete[] mmData;
	return result;
}

Tensor Tensor::transpose() const {
	Tensor result;
	double *transposeData = new double[ totalElements( size ) ];
	transposeMatrix( std::get<0>( size ), std::get<1>( size ), data, transposeData ); 
	if( autograd ) {
		const Tensor* const *c = new const Tensor* const[1]{ this };
		result = Tensor{ std::tuple<int, int>{std::get<1>( size ), std::get<0>( size)}, transposeData, true, c, TRANSPOSE };
		delete[] c;
	} else {
		result = Tensor{ size, transposeData };
	}

	delete[] transposeData;
	
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
				case SUB:
					creators[0]->backward( grad, this );
					creators[1]->backward( -grad, this );
					break;
				case MUL:
					creators[0]->backward( grad * *creators[1], this );
					creators[1]->backward( grad * *creators[0], this );
					break;
				case NEG:
					creators[0]->backward( -grad );
					break;
				case MM:
					creators[0]->backward( this->grad->mm( creators[1]->transpose()) );
					creators[1]->backward( this->grad->transpose().mm( *creators[0] ).transpose() );
					break;
				case TRANSPOSE:
					creators[0]->backward( this->grad->transpose() );
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
		for( int i=1; i<totalElements(size); i++ ) {
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
		case SUB:
		case MUL:
		case MM:
			creators[0]->addChild( id );
			creators[1]->addChild( id );
			break;
		case NEG:
		case TRANSPOSE:
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

