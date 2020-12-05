# Binarized Neural Network 
<a href="https://www.dis.uniroma1.it/"><img src="http://www.dis.uniroma1.it/sites/default/files/marchio%20logo%20eng%20jpg.jpg" width="500"></a>

Tensorflow framework and Python_3 are required; we trained the network over 2 types of dataset: MNIST and CIFAR-10

For each dataset we will implement: 
  1) A binarized neural network with original batch normalizzation 
  2) A binarized neural network with shift based batch normalizzation.

The default optimizer is the Vanilla Adam, also we improved the network with a particular extension optimization, the Shift Based AdaMax. 

In the end it's possible to plot graphics about accuracy and loss of the trained network.

# Documentation
 See the [report](./Report.pdf)

# Cifar dataset
![](cifar.png)

# MNIST dataset
![](mnist.png)


# Partial Results (sbn adamax)
![](risultati-sbn-adamax/a.png)
![](risultati-sbn-adamax/b.png)


All results are in the report
