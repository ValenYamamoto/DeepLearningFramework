#ifndef TENSOR_H 
#define TENSOR_H
#include <vector>
#include <map>
#include <string>


class Tensor {
	public:
		enum CreationOp {
			ADD, 
			NEG,
			MUL,
			SUM,
			EXPAND,
			TRANSPOSE,
			MM,
			NONE
		};

		Tensor();
		
		Tensor( std::vector<double> values, bool autograd=false, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE, int id=-1 );


		Tensor( const Tensor& original );

		~Tensor();

		Tensor getGrad();

		void backward( Tensor grad, const Tensor* gradOrigin=nullptr ) const;

		Tensor operator +( const Tensor &right );

		Tensor operator -();
		
		Tensor& operator =( const Tensor &right );

		std::string to_string() const;

		bool getAutograd() const;

	private:
		long unsigned int size;
		mutable Tensor *grad;
		double *data;
		CreationOp creationOp;
		const Tensor* const *creators;
		bool autograd;
		int id;
		mutable std::map<int, int> children;

		static int nextID;


		Tensor( long unsigned int size, double *data, bool autograd=false, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE, int id=-1 );

		void addChild( int id ) const;
		void createChildren() const;
		static int createTensorId();
		bool gradFromAllChildren() const;
};
#endif
