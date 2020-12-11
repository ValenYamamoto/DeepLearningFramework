#include <functional>
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
			case Tensor::SUM:
				return new const Tensor* const[1]{ creators[0] };
			default:
				return nullptr;	
		}
	}

	void naiveMatrixMult( int m, int n, int o, double *m1, double *m2, double *result ) {
		int i, j, k;
		for( i=0; i<m; i++ ) {
			for( j=0; j<o; j++ ) {
				*( result+o*i+j ) = 0;
				for( k=0; k<n; k++ ) {
					*( result+o*i+j ) += *( m1+n*i+k ) * *( m2+o*k+j );
				}
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

	int totalElements( std::tuple<int, int> sz ) {
		return std::get<0>( sz ) * std::get<1>( sz );
	}

	void sumDimZero( std::tuple<int, int> sz, double *m, double *result ) {
		int i;
		for( i=0; i< std::get<1>( sz ); i++ ) {
			double sum = 0;
			for( int j=0; j< std::get<0>( sz ); j++ ) {
				sum += *(m + std::get<1>(sz) * j + i );
			}
			*(result+i) = sum;
		}
	}
}

int Tensor::nextID = 0;
bool Tensor::noGrad = false;

Tensor::Tensor() {
	#if DEBUG
	std::cout << "No arg Constructor" << std::endl;
	#endif
	sz = std::tuple<int, int>{ 1, 1 };
	data = nullptr;
	creators = nullptr;
	grad = nullptr;
	creationOp=NONE;
	autograd = false;
	backpropFunc = []( double d ){ return d; };
}

Tensor::Tensor( std::tuple<int, int> size, double *data, bool autograd, const Tensor* const creators[], CreationOp creationOp ) : sz{size}, grad{nullptr}, data{new double[totalElements(size)] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Array Constructor" << std::endl;
	#endif
	for( int i=0; i<totalElements(sz); i++ ) {
		this->data[i] = data[i];
	}
	if( creators != nullptr ) {
		createChildren();
	}
	backpropFunc = []( double d ){ return d; };
}

Tensor::Tensor( std::vector<double> values, bool autograd, const Tensor* const creators[], CreationOp creationOp ) : sz{std::tuple<int, int>(values.size(), 1 )}, grad{nullptr}, data{new double[totalElements(sz)] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
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
	backpropFunc = []( double d ){ return d; };
}

Tensor::Tensor( std::vector<std::vector<double>> values, bool autograd, const Tensor* const creators[], CreationOp creationOp ) : sz{std::tuple<int, int>(values.size(), values[0].size() )}, grad{nullptr}, data{new double[totalElements(sz)] }, creationOp{creationOp}, creators{ createCreators(creators, creationOp )}, autograd{autograd}, id{Tensor::createTensorId()} {
	#if DEBUG
	std::cout << "Vector Constructor" << std::endl;
	#endif
	std::vector<std::vector<double>>::iterator rowPointer = values.begin();
	for( int i=0; rowPointer<values.end(); i++, rowPointer++ ) {
		std::vector<double>::iterator dataPointer = rowPointer->begin();
		for( int j=0; dataPointer<rowPointer->end(); j++, dataPointer++ ) {
			data[ std::get<1>( sz ) * i + j ] = *dataPointer;
		}
	}
	if( creators != nullptr ) {
		createChildren();
	}
	backpropFunc = []( double d ){ return d; };
}

Tensor::Tensor( const Tensor& original ) : sz{original.sz}, data{new double[totalElements(sz)]}, creationOp{original.creationOp}, creators{createCreators(original.creators, creationOp) }, autograd{original.autograd}, id{original.id} {
	#if DEBUG
	std::cout << "Copy Constructor" << std::endl;
	#endif
	for( int i=0; i<totalElements(sz); i++ ) {
		data[i] = original.data[i];
	}
	if( original.grad == nullptr ) {
		grad = nullptr;
	} else {
		grad = new Tensor(*original.grad);
	}
	if( creators != nullptr ) {
		createChildren();
	}
	backpropFunc = original.backpropFunc;
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

Tensor Tensor::random( int rows, int cols, bool autograd ) {
	std::tuple<int, int> sz = std::tuple<int, int>{ rows, cols };
	double *data = new double[ rows * cols ];
	int i;
	for( i=0; i<rows*cols; i++ ) {
		data[i] = 2 * ( std::rand() / (double) RAND_MAX ) - 1;
	}
	Tensor result = Tensor{ sz, data, autograd };
	delete[] data;
	return result;
}

Tensor Tensor::fill( int rows, int cols, double value, bool autograd ) {
	std::tuple<int, int> sz = std::tuple<int, int>{ rows, cols };
	double *data = new double[ rows * cols ];
	int i;
	for( i=0; i<rows*cols; i++ ) {
		data[i] = value;
	}
	Tensor result = Tensor{ sz, data, autograd };
	delete[] data;
	return result;
}

Tensor& Tensor::operator =( const Tensor &right ) {
	#if DEBUG
	std::cout << "= operator Constructor" << std::endl;
	#endif
	if( this != &right ) {
		if( data != nullptr ) {
			delete[] data;
		}
		data = new double[totalElements(right.sz)];
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
		
		this->sz = right.sz;
		int i;
		for( i=0; i<totalElements(sz); i++ ) {
			this->data[i] = right.data[i];
		}
		creationOp = right.creationOp;
		creators = createCreators( right.creators, creationOp );
		autograd = right.autograd;
		id = right.id;
		backpropFunc = right.backpropFunc;
	}
	return *this;
}

Tensor Tensor::operator +( const Tensor &right ) {
	double *addData = new double[totalElements(right.sz)];

	int i;
	for( i=0; i<totalElements( sz ); i++ ) {
		addData[i] = right.data[i] + this->data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd && !noGrad ) {
		result = Tensor{ sz, addData, true, c, ADD };
	} else {
		result = Tensor{ sz, addData };
	}

	delete[] addData;
	delete[] c;

	return result;
}

Tensor Tensor::operator -() {
	Tensor result;
	double *negateData = new double[ totalElements( sz ) ];
	int i;
	for( i=0; i<totalElements( sz ); i++ ) {
		negateData[i] = data[i] * -1;
	}
	if( autograd && !noGrad ) {
		const Tensor* const *c = new const Tensor* const[1]{ this };
		result = Tensor{ sz, negateData, true, c, NEG };
		delete[] c;
	} else {
		result = Tensor{ sz, negateData };
	}

	delete[] negateData;
	
	return result;
}

Tensor Tensor::operator -( const Tensor &right ) {
	double *addData = new double[totalElements(right.sz)];

	int i;
	for( i=0; i<totalElements( sz ); i++ ) {
		addData[i] = this->data[i] - right.data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd && !noGrad ) {
		result = Tensor{ sz, addData, true, c, SUB };
	} else {
		result = Tensor{ sz, addData };
	}

	delete[] addData;
	delete[] c;

	return result;
}

Tensor Tensor::operator *( const Tensor &right ) {
	double *mulData = new double[totalElements(right.sz)];

	int i;
	for( i=0; i<totalElements(sz); i++ ) {
		mulData[i] = this->data[i] * right.data[i];
	}

	const Tensor* const *c = new const Tensor* const[2]{ this, &right };

	Tensor result;
	if( autograd && right.autograd && !noGrad ) {
		result = Tensor{ sz, mulData, true, c, MUL };
	} else {
		result = Tensor{ sz, mulData };
	}

	delete[] mulData;
	delete[] c;

	return result;
}

Tensor Tensor::operator *( const double &right ) {
	double *mulData = new double[ totalElements( sz ) ];
	int i;
	for( i=0; i<totalElements( sz ); i++ ) {
		mulData[i] = data[i] * right;
	}

	Tensor result{ sz, mulData };
	delete[] mulData;
	return result;
}

Tensor Tensor::mm( const Tensor &right ) {
	double *mmData = new double[ std::get<0>( sz ) * std::get<1>( right.sz ) ];

	naiveMatrixMult( std::get<0>( sz ), std::get<1>( sz ), std::get<1>( right.sz ), data, right.data, mmData );

	Tensor result;
	if( autograd && right.autograd && !noGrad ) {
		const Tensor* const *c = new const Tensor* const[2]{ this, &right };
		result = Tensor{ std::tuple<int, int>{ std::get<0>( sz ), std::get<1>( right.sz )}, mmData, true, c, MM }; 
		delete[] c;
	} else {
		result = Tensor{ std::tuple<int, int>{ std::get<0>( sz ), std::get<1>( right.sz )}, mmData };
	}

	delete[] mmData;
	return result;
}

Tensor Tensor::transpose() const {
	Tensor result;
	double *transposeData = new double[ totalElements( sz ) ];
	transposeMatrix( std::get<0>( sz ), std::get<1>( sz ), data, transposeData ); 
	if( autograd && !noGrad ) {
		const Tensor* const *c = new const Tensor* const[1]{ this };
		result = Tensor{ std::tuple<int, int>{std::get<1>( sz ), std::get<0>( sz)}, transposeData, true, c, TRANSPOSE };
		delete[] c;
	} else {
		result = Tensor{ std::tuple<int, int>{std::get<1>( sz ), std::get<0>( sz )}, transposeData };
	}

	delete[] transposeData;
	
	return result;
}

Tensor Tensor::sum( int dim ) {
	if( dim == 0 ) {
		double *sumData = new double[ std::get<0>( sz ) ];
		sumDimZero( sz, data, sumData );
		Tensor result;
		if( autograd ) {
			const Tensor* const *c = new const Tensor* const[1]{ this };
			result = Tensor{ std::tuple<int, int>{ 1, std::get<1>( sz )}, sumData, true, c, SUM };
			delete[] c;
		} else {
			result = Tensor{ std::tuple<int, int>{ 1, std::get<1>( sz ) }, sumData };
		}
		result.dim = dim;
		delete[] sumData;
		return result;
	}
	return Tensor();
}

Tensor Tensor::expand( int dim, int copies ) {
	Tensor result;
	double *expandData = new double[ totalElements( sz ) * copies ];
	int i;
	for( i=0; i<totalElements( sz ); i++ ) {
		int j;
		for( j=0; j<copies; j++ ) {
			expandData[ i+totalElements( sz )*j ] = data[i];
		}
	}
	result = Tensor{ std::tuple<int, int>{ copies, std::get<1>( sz ) }, expandData };
	delete[] expandData;
	return result;
}

double& Tensor::operator[]( int index ) {
	return data[ index ];
}

void Tensor::backward( Tensor grad, const Tensor* gradOrigin ) const {
	if( autograd && !noGrad ) {
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
			for( int i=0; i<totalElements( this->grad->sz ); i++ ) {
				this->grad->data[i] = backpropFunc( this->grad->data[i] );
			}
			int ds;
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
				case SUM:
					ds = std::get<0>( creators[0]->sz );
					creators[0]->backward( this->grad->expand( dim, ds ) );
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
		std::string s = "";
		s += "<" + std::to_string( data[0] );
		for( int i=1; i<totalElements(sz); i++ ) {
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
		case SUM:
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
			return false;
		}
	}
	return true;
}

bool Tensor::getAutograd() const {
	return autograd;
}

void Tensor::setNoGrad( bool b ) {
	noGrad = b;
}

void Tensor::update( const Tensor &change ) {
	int i;
	for( i=0; i < totalElements( sz ); i++ ) {
		data[i] -= change.data[i];
	}
}

void Tensor::clearGrad() {
	if( grad != nullptr ) {
		delete grad;
		grad = nullptr;
	}
}

std::tuple<int, int> Tensor::size() {
	return sz;
}

double* Tensor::getData() {
	return data;
}

void Tensor::setBackpropFunction( BackpropFunction func ) {
	backpropFunc = func;
}

void Tensor::applyActivationFunction( BackpropFunction func ) {
	for( int i=0; i<totalElements( sz ); i++ ) {
		data[i] = func( data[i] );
	}
}
