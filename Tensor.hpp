#ifndef TENSOR_H 
#define TENSOR_H
#include <vector>
#include <tuple>
#include <map>
#include <string>


class Tensor {
	public:
		enum CreationOp {
			ADD, 
			SUB,
			NEG,
			MUL,
			SUM,
			EXPAND,
			TRANSPOSE,
			MM,
			NONE
		};

		static bool noGrad;

		Tensor();

		Tensor( std::vector<std::vector<double>> values, bool autograd=false, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE, int id=-1 );

		Tensor( std::vector<double> values, bool autograd=false, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE, int id=-1 );

		Tensor( std::tuple<int, int> size, double *data, bool autograd=false, const Tensor* const creators[]=nullptr, CreationOp creationOp=NONE, int id=-1 );

		Tensor( const Tensor& original );

		~Tensor();

		Tensor getGrad();

		void backward( Tensor grad, const Tensor* gradOrigin=nullptr ) const;

		Tensor operator +( const Tensor &right );

		Tensor operator -();

		Tensor operator -( const Tensor &right );

		Tensor operator *( const Tensor &right );
		
		Tensor operator *( const double &right );
		
		Tensor mm( const Tensor &right );
		
		Tensor transpose() const;

		Tensor sum( int dim=0 );
		
		Tensor expand( int dim, int copies ); 

		Tensor& operator =( const Tensor &right );

		std::string to_string() const;

		bool getAutograd() const;

		static Tensor random( int rows, int cols, bool autograd=false );

		static Tensor fill( int rows, int cols, double value, bool autograd=false );

		static void setNoGrad( bool b );

		void update( const Tensor &change );

		void clearGrad();

	private:
		std::tuple<int, int> size;
		mutable Tensor *grad;
		double *data;
		CreationOp creationOp;
		const Tensor* const *creators;
		bool autograd;
		int id;
		mutable std::map<int, int> children;

		static int nextID;

		int dim;


		void addChild( int id ) const;
		void createChildren() const;
		static int createTensorId();
		bool gradFromAllChildren() const;
};
#endif
