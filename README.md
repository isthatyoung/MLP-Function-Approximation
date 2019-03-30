# MLP-Function-Approximation
An implementation of multilayer perceptron(MLP) on function approximation.

## Requirement
Python 3.7  
torch 1.0.0  
Numpy 1.15.2  
Matplotlib 1.3.1  

## Description
This Python script implement the backpropagation algorithm for a 1 - S - 1 architecture network, S means the numbers of neurons in hidden layer. We use torch tensor to realize matrix operation within the forward and backward propogation. The weights and biases are randomly initialized and uniformly distributed between -0.5 and 0.5 (using the function rand).

## Example approximation function
<img src="http://www.forkosh.com/mathtex.cgi? g(p) = 0.1 * p^{2} * sin(p) for -3\leq p \leq 3">  


## Result
![enter image description here](https://github.com/isthatyoung/MLP-Function-Approximation/blob/master/result/approximate_function.png)  

|Neurons |Learning rate | Epochs of training| 
|------|----------|---|
|100|0.01|1000|