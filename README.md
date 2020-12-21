# Deep Learning Framework C++
#### HKN Initiation Project, Fall 2020

## Why This Project?
After a entire quarter doing coding projects for class, I want to do something that I'm interested. Since I've spent this quarter's free time looking into deep learning, it seemed like a good time to spend some time looking into the math behind deep learning.
Many jobs ask for profiency in C++; since I am taking a class this quarter in C++ might as well kill two birds with one stone and spend some time mucking around in C++. I tried to hit all the important topics:
* Classes
* Polymorphism and Inheritance
* Operator Overloading
* Standard C++ containers and iterators
* Functions/Lambdas
* Templates

Besides just mastery of the language, I also tried to get a handle on important tools used when writing C++ programs:
* Makefiles
* gdb
* valgrind memory check

## Overview and Project Goals
* Deep learning framework loosely based off PyTorch
	* Automatic Backpropagation and optimization
	* Dynamic Computation graphs
* Design is modular and easy to extend
	* Layers, optimizers inherit from abstract base classes for common interface, overload functions to create different effects

## Automatic Differentiation and Dynamic Computation Graphs
* Created own Tensor class
	* handles all backpropation logic: each Tensor knows its children and its parents
* Computation graph is dynamically created in Tensor class via operator overloading arithmetic functions (plus a few extras)
	* Scalar addition, multiplication, subtraction; matrix multiplication; sum along dimension
* As data moves forward through network, Tensors are created and added to graph; calling backward with an error on the last result Tensor backpropagates the error backwards through the network

## Layers
* Layers are implemented by inheriting from the abstract base class Layer.hpp, which defines the function forward, which all implementaions must implement themselves, and a getParamters function, which is the same for every layer regardless of type
* Have single layers and containers:
	* Linear
	* LinearWithFunction - linear layer with activation function; activation function and derivative passed in as parameters when layer is instantiated
	* Sequential - container that holds a list of layers and propagates signals through them in order

## Next Steps
* Expand!
	* more layers, more algorithms
	* Maybe some CUDA?
