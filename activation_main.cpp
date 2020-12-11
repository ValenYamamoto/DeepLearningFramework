#include <tuple>
#include <iostream>
#include "Tensor.hpp"
#include "SGD.hpp"
#include "./layers/Layer.hpp"
#include "./layers/LinearWithFunction.hpp"
#include "./layers/Sequential.hpp"


double relu( double d ) {
	return (d > 0) * d;
}

double reluDeriv( double d ) {
	return (d>0);
}

int main() {
	double d[8] = {0,0,0,1,1,0,1,1};
	double t[4] = {0, 1, 0, 1};


	LinearWithFunction layer1 = LinearWithFunction( 2, 3, relu, reluDeriv );
	LinearWithFunction layer2 = LinearWithFunction( 3, 1, relu, reluDeriv );
	Layer* layers[2] = { &layer1, &layer2 };
	Sequential model = Sequential( 2, layers ); 

	Tensor data = Tensor( std::tuple<int, int>{4, 2}, d, true );
	Tensor target = Tensor( std::tuple<int, int>{4, 1}, t, true );

	SGD optim = SGD( 2, model.getParameters(), 0.1 );

	for( int i=0; i < 10; i++ ) {

			Tensor prediction = model.forward( data );
			std::cout << "prediction is: " << prediction.to_string() << std::endl;
			std::cout << "target is: " << target.to_string() << std::endl;
			Tensor delta = prediction - target;

			Tensor loss = delta * delta;
			Tensor lossSum = loss.sum();
			lossSum.backward( Tensor{ std::vector<double>{ 1 } } );


			optim.step( false );
			optim.zero();

			std::cout << "Loss is: " << lossSum.to_string() << std::endl << std::endl;

	}
}
