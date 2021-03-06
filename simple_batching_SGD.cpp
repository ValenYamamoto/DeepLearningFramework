#include <tuple>
#include <iostream>
#include "Tensor.hpp"
#include "SGD.hpp"

int main() {
	double d[8] = {0,0,0,1,1,0,1,1};
	double t[4] = {0, 1, 0, 1};

	Tensor weight1 = Tensor::random( 2, 3, true );
	Tensor weight2 = Tensor::random( 3, 1, true );

	Tensor data = Tensor( std::tuple<int, int>{4, 2}, d, true );
	Tensor target = Tensor( std::tuple<int, int>{4, 1}, t, true );

	Tensor* weights[2] = { &weight1, &weight2 };
	SGD optim = SGD( 2, weights, 0.1 );

	for( int i=0; i < 10; i++ ) {

			Tensor hidden = data.mm( weight1 );

			std::cout << "hidden is: " << hidden.to_string() << std::endl;
			
			Tensor prediction = hidden.mm( weight2 );
			std::cout << "prediction is: " << prediction.to_string() << std::endl;
			std::cout << "target is: " << target.to_string() << std::endl;
			Tensor delta = prediction - target;
			// std::cout << "delta is: " << delta.to_string() << std::endl;

			Tensor loss = delta * delta;
			Tensor lossSum = loss.sum();
			lossSum.backward( Tensor{ std::vector<double>{ 1 } } );

			std::cout << "weights before: " << weight1.to_string() << " " << weight2.to_string() << std::endl;
			std::cout << "weight grads: " << weight1.getGrad().to_string() << " "<< weight2.getGrad().to_string() << std::endl;

			optim.step( false );
			optim.zero();

			std::cout << "weights after: " << weight1.to_string() << " " << weight2.to_string() << std::endl;

			std::cout << "Loss is: " << lossSum.to_string() << std::endl << std::endl;

	}
}
