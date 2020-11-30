#include <iostream>
#include <vector>
#include "Tensor.hpp"

int main() {
	std::cout << "test start" << std::endl;
	std::cout << "a" << std::endl;
	Tensor a = Tensor( std::vector<double>{ 1, 2, 3, 4 } );
	std::cout << "b" << std::endl;
	Tensor b = Tensor( std::vector<double>{ 5, 6, 7, 8 } );

	std::cout << "c" << std::endl;
	Tensor c = a + b;

	std::cout << "to string" << std::endl;
	std::cout << a.to_string() << std::endl;
	std::cout << b.to_string() << std::endl;
	std::cout << c.to_string() << std::endl;
	std::cout << "backward" << std::endl;
	c.backward( Tensor( std::vector<double>{ 1, 1, 1, 1 } ) );
	std::cout << "getGrad" << std::endl;
	std::cout << a.getGrad().to_string() << std::endl;
	std::cout << b.getGrad().to_string() << std::endl;
	return 0;
}
