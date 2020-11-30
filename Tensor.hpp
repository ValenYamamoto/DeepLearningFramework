#ifndef TENSOR_H 
#define TENSOR_H
#include <vector>
#include <string>


class Tensor {
	public:
		enum CreationOp {
			ADD, 
			SUB,
			MUL,
			SUM,
			EXPAND,
			TRANSPOSE,
			MM,
			NONE
		};

		Tensor();
		
		Tensor( std::vector<double> values, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE );


		Tensor( const Tensor& original );

		~Tensor();

		Tensor getGrad();

		void backward( Tensor grad ) const;

		Tensor operator +( const Tensor &right );
		
		Tensor& operator =( const Tensor &right );

		std::string to_string();

	private:
		long unsigned int size;
		mutable Tensor *grad;
		double *data;
		CreationOp creationOp;
		const Tensor* const *creators;

		Tensor( long unsigned int size, double *data, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE );
};
#endif
