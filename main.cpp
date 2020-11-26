#include <iostream>
#include <vector>
#include "Tensor.hpp"

int main() {
	Tensor a = Tensor( std::vector<double>{ 1, 2, 3, 4 } );
	Tensor b = Tensor( std::vector<double>{ 5, 6, 7, 8 } );

	Tensor c = a + b;

	std::cout << a.to_string() << std::endl;
	std::cout << b.to_string() << std::endl;
	std::cout << c.to_string() << std::endl;
	c.backward( Tensor( std::vector<double>{ 1, 1, 1, 1 } ) );
	std::cout << a.getGrad().to_string() << std::endl;
	std::cout << b.getGrad().to_string() << std::endl;
	return 0;
}
