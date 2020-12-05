#include <iostream>
#include <vector>
#include "Tensor.hpp"

int main() {
	std::cout << "test start" << std::endl;
	Tensor a = Tensor( std::vector<double>{ 1, 2, 3, 4, 5 }, true );
	Tensor b = Tensor( std::vector<double>{ 2, 2, 2, 2, 2 }, true );
	Tensor c = Tensor( std::vector<double>{ 5, 4, 3, 2, 1 }, true );

	Tensor bb = -b;
	Tensor bbb = -b;
	Tensor d = a + bb;
	Tensor e = bbb + c;
	Tensor f = d + e;


	std::cout << "to string" << std::endl;
	std::cout << a.to_string() << std::endl;
	std::cout << b.to_string() << std::endl;
	std::cout << c.to_string() << std::endl;
	std::cout << d.to_string() << std::endl;
	std::cout << e.to_string() << std::endl;
	std::cout << f.to_string() << std::endl;
	std::cout << "backward" << std::endl;
	f.backward( Tensor( std::vector<double>{ 1, 1, 1, 1 } ) );
	std::cout << "getGrad" << std::endl;
	std::cout << a.getGrad().to_string() << std::endl;
	std::cout << b.getGrad().to_string() << std::endl;

	Tensor t1{ std::vector<double>{ 1, 2, 3, 4, 5 } };
	Tensor t2 = t1.transpose();
	std::cout << t2.to_string() << std::endl;
	t2 = t1.mm( t2 );
	std::cout << t2.to_string() << std::endl;


	Tensor t3{ std::vector<std::vector<double>>{ std::vector<double>{ 1, 2, 3 }, std::vector<double>{ 4, 5, 6 } } };
	std::cout << t3.to_string() << std::endl;
	Tensor t4{ std::vector<std::vector<double>>{ std::vector<double>{ 1, 2 }, std::vector<double>{ 3, 4 } } };
	Tensor t5 = t4.mm( t4 );
	std::cout << t5.to_string() << std::endl;

	return 0;
}
