#include "Tensor.hpp"

class SGD {
	public:
		SGD( int numParam, Tensor** parameters, double alpha=0.01 );
		~SGD();
		void zero();
		void step( bool zero=true );
	private:
		int numParam;
		Tensor** parameters;
		double alpha;
};
