#include <iostream>
#include <vector>
#include "Tensor.hpp"

int main() {
	std::cout << "test start" << std::endl;
	std::cout << "a" << std::endl;
	Tensor a = Tensor( std::vector<double>{ 1, 2, 3, 4 }, true );
	std::cout << "b" << std::endl;
	Tensor b = Tensor( std::vector<double>{ 5, 6, 7, 8 }, true );
	std::cout << "A autograd " << a.getAutograd() << std::endl;
	std::cout << "B autograd " << b.getAutograd() << std::endl;

	std::cout << "c" << std::endl;
	Tensor c = a + b;
	std::cout << "C autograd " << c.getAutograd() << std::endl;
	std::cout << "A autograd " << a.getAutograd() << std::endl;
	std::cout << "B autograd " << b.getAutograd() << std::endl;

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
