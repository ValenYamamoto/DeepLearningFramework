all: exe obj exe/activation.out

exe :
	mkdir $@

obj :
	mkdir $@

exe/sequential.out : sequential_main.cpp obj/Sequential.o obj/Linear.o obj/Tensor.o obj/SGD.o
	g++ -Wall -g -o exe/sequential.out sequential_main.cpp obj/Linear.o obj/Sequential.o obj/Tensor.o obj/SGD.o

exe/activation.out : activation_main.cpp obj/Sequential.o obj/LinearWithFunction.o obj/Tensor.o obj/SGD.o
	g++ -Wall -g -o exe/activation.out activation_main.cpp obj/LinearWithFunction.o obj/Sequential.o obj/Tensor.o obj/SGD.o

obj/Tensor.o : Tensor.cpp Tensor.hpp
	g++ -Wall -g -c -o obj/Tensor.o Tensor.cpp

obj/SGD.o : SGD.cpp SGD.hpp
	g++ -Wall -g -c -o obj/SGD.o SGD.cpp

obj/Sequential.o : layers/Sequential.cpp layers/Sequential.hpp layers/Layer.hpp
	g++ -Wall -g -c -o obj/Sequential.o layers/Sequential.cpp

obj/Linear.o : layers/Linear.cpp layers/Linear.hpp layers/Layer.hpp
	g++ -Wall -g -c -o obj/Linear.o layers/Linear.cpp

obj/LinearWithFunction.o : layers/LinearWithFunction.cpp layers/LinearWithFunction.hpp layers/Layer.hpp
	g++ -Wall -g -c -o obj/LinearWithFunction.o layers/LinearWithFunction.cpp
