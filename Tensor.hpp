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
		
		template <typename Iter>
		explicit Tensor( Iter values, Tensor *creators=nullptr, CreationOp creationOp=NONE );
		Tensor( int size, double *data, Tensor *creators=nullptr, CreationOp creationOp=NONE );


		Tensor( const Tensor& original );

		~Tensor();

		Tensor getGrad();

		void backward( Tensor grad );

		Tensor operator +( const Tensor &right );
		
		Tensor& operator =( const Tensor &right );

		std::string to_string();

	private:
		int size;
		Tensor *grad;
		double *data;
		Tensor *creators;
		CreationOp creationOp;

		void createCreators( Tensor *creators, CreationOp creationOp );
};
#endif
