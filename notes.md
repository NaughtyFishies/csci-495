# Notes
## Week 1-2
### Feedforward Neural Networks
* Information moves in only 1 direction, from the input nodes through the hidden nodes (if any) and to the output nodes.
* There are no cycles or loops in the network.
#### Architecture
* Each layer is made up of units called neurons and the layers are interconnected by weights.
* The input layer
    + This layer consists fo neurons that recieve inputs and pass them to the next layer. The number of neurons in hte input layer is determined by the dimensions of the input data.
* The hidden layers
    + These layers are not exposed to the input or output and can be considered the computational engien of the neural network.
    + Each hidden layer's neurons take the weighted sum of the outputs from the previous layer, apply an **activation function**, and pass the result to the next layer.
    + The network can have zero or more hidden layers.
* The output layer
    + This final layer produces the output for the given inputs.
    + The number of neurons in the output layer depends on the number of possible outputs the network is designed to produce.
* Each neuron on one layer is connected to every neuron in the next layer.
* The strength of the connection between neurons is represented by weights.
* Learning in a neural network involves updating these weights based on the error of the output.
#### How do they work?
##### Feedforward Phase
* Input data is fed into the network, propogating forward.
* At each hidden layer, the weighted sum of the inputs is calculated and passsed through and activation function which introduces non-linearity into the model.
* This process continues until the output layer and a prediction is made.
##### Backpropagation Phase
* Once a prediction is made, the error (difference between the predicted output and actual output) is calculated.
* This error is then propogated back through the network, and the weights are adjusted to minimize this error.
* The process of adjusting weights is typically doen using a **gradient descent optimization algorithm**.
#### Training Feedforward Neural Networks / Gradient Descent
* Involves using a dataset to adjust the weights of the connections between neurons.
* This is done by passing the dataset through the network multiple times and updating the weights each time to reduce the error in prediction.
* The process is called gradient descent and it continues until the network perfroms satisfactorily on the training data.
#### Applications of Feedforward Neural Networks
* Pattern recognition
* Classification tasks
* Regression analysis
* Image recognition
* Time series prediction
#### Challenges and Limitiations
* How to decide how many hidden layers to add to the neural network.
    + This can effect the performance of the network significantly.
* Overfitting
    + If you train the network too well on the training data then it performs poorly on new unseen data.
#### Sources
[Deep AI Feedforward Neural Networks](https://deepai.org/machine-learning-glossary-and-terms/feed-forward-neural-network)

### Activation Functions
* A function used in neural networks which outputs a small value for small inputs and a larger value if its inputs exceed a threshold.
* If the inputs are large enough, the activation function "fires".
* In other words, an activation function is like a gate that checks that an incoming value is greater than a critical number.
* They add non-linearity to neural networks allowing the networks to learn powerful operations.
* Without activation functions the entire network could be re-factored to a simple linear operationor matrix transformation on its input and would no longer be capable of performing complex tasks like image recognition.
* Well-known activation functions include the **rectified linear unit (ReLU)** function and the family of **sigmoid functions** (logistic sigmoid function, the hyperbolic tangent and the arctangent function).

#### ReLU
* The rectified linear unit or ReLU is a piecewise linear function that outputs zero if the input is negative and directly outputs the input otherwise.
* When negative x, the derivitive is 0 which can cause problems when training since a neuron can become "trapped" in the zero region adn backpropogation will never change its weights.

#### PReLU
* Due to the zero gradient problem faced by the ReLU, its fairly common to use the PReLU.
* This has the advantage that the gradient is nonzero at all points (except at 0 where it is undefined).
* The PReLU function with $\alpha$ set to 0.1 avoids the zero gradient problem.

#### Logistic Sigmoid Function Formula
* The logistic sigmoid function has its gradient defined everywhere and all output is between 0 and 1 for all x.
* The logistic sigmoid function is easier to work with mathematically but the exponential functions make it computationally intensive to compute in practice so simpler functions like the ReLU are often preferred.
* The derivative of the logistic sigmoid function is nonzero at all points which is an advantage for use in the backpropogation algorithm although the function is intense to compute and the gradient becomes very small for large absolute x giving rise to the **vanishing gradient problem**.
* Because the derivative contains exponentials, it is compuationally expensive to calculate.
* Backpropogation requires the derivative of all operations in a neural network to be calculated so the sigmoid function is not well suited for use in neural networks in practice due to the complexity of calculating its deriviative repeatedly.

#### Sources
[Deep AI Activation Functions](https://deepai.org/machine-learning-glossary-and-terms/activation-function)