# MIT Intro Notes
[Source](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
### Why deep learning?
* Hand engineered features are time consuming and brittle and not scalable
* Data is more abundent that ever in human history, otherwise these models would not be possible.
* GPUs are able to compute in parallel which is necessary for the amount of computation a deep learning model requires.
### The Perceptron - The structural building block of deep learning
* A perceptron is a neuron in a neural network
* Inputs are multiplied by their weights which get summed at the neuron.
* The sum is multiplied by a non-linear activation function to achieve the output.
* A bias term is also included as an input which allow the mathematical function to "shift" on the x-axis.
* We can think of these terms as vectors so **X** is the vector of all inputs and **W** is the list of weights.
* The input is obtained by taking the dot product of **X** and **W**, then add the bias term at the end. 
* We can call the non-linearity g, and the bias term w $_{0}$ and the output $\hat{y}$, we get the equation $\hat{y}$ = g (w $_{0}$ + **X** $\cdot$ **W**)
* Say we have w $_{0}$ = 1 and **W** = $\begin{bmatrix}
   3 \\
   -2
\end{bmatrix}$, then our $\hat{y}$ would be a line in 2D, so we can plot it
* So we can see how this neuron is working and performing and how it seperates data.
* The activation function seperates the input into two different section.
* This graph of a neuron's line is called a feature space and allows us to visualize how exactly this neuron is performing in its completion.
#### Activation Functions
* Common actvation function is sigmoid function
    + Squashes every input into a value between 0-1.
* ReLU function
    + linear at all points except at x=0.
    + very fast, two linear function piecewise combined together
* What is the point of these activation functions
    + These allow our models to work with data that is non-linear
    + By adding non-linearities they allow the models to be more powerful, smaller and more expressive.
#### Building Neural Networks with Perceptrons
* Each perceptron recieves its own "line" so it has its own weight.
* The basic steps for each neuron are:
    + Take the dot product of inputs and weights
    + Add the bias term
    + Apply a non-linearity
```python
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init()

        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward propogate the inputs
        # Take the dot product of inputs and weight and add bias
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output
```
* All of the layer tools and architecture is provided for you in tools like pytorch and tensor flow

##### Initializing a 2 neuron dense layer
```python
import tensorflow as tf

layer = tf.keras.layers.Dense(units=2)
```

* Layers between input and output are called hidden layers
* In this case you would have a weight matrix for the first connects **$W^{(1)}$** and a second one for the next set of connections **$W^{(2)}$**
* Need cascading non-linearities or else it will be a linear output

##### Intializing 2 dense layers
```python
import tensorflow as tf

model = tf.keras.Sequential([
        tf.keras.layers.Dense(n),
        tf.keras.layers.Dense(2)
])
```
* Non-linearity function does not need to be the same between sequential layers but often it is for convience
* A deep neural network is just many layers stacked on top of each other
##### Intializing a deep neural network
```python
import tensorflow as tf

model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_1),
        tf.keras.layers.Dense(n_2),
        .
        .
        .
        tf.keras.layers.Dense(2)
])
```
### Applying Neural Networks
#### Example Problem: Will I pass this Class
* Two feature model
    + $x_{1}$ = # of lectures you attend
    + $x_{2}$ = Hours spent on the final project

* Observe data showing how people how done in the past given these variables
* You are at $\begin{bmatrix}
   4 \\
   5
\end{bmatrix}$ 
so you have attended 4 lectures and spent 5 hours on the final project
* You have a predicted chance of 0.1 or 10% but the actual is 1 or 100% so what went wrong? **The model is not trained**
* Compute loss which shows the difference between predicted and actual
* Want to minimize the empirical loss on average across all data points
```python
loss = tf.reduce_mean(tf.square(tf.subtract (y, predicted)))
loss = tf.keras.losses.MSE(y, predicted)
```
### Training Neural Network
#### Loss Optimization
We want to find the network weights that achieve the lowest loss \
* $W^{*}$ = argminJ(**W**)
* Remember **W** = {$W^{(0)}$, $W^{(1)}$, ...}
* Can plot loss landscape over 2D space and want to find lowest point, this is where our weights are the best so our loss is the lowest
* Start somewhere random in the place and compute the gradient of the landscape at that place
* Take a small step going down and repeat the process continously until we found a local minimum
* This algorithm is called gradient descent
#### Gradient Descent algorithm:
* Intialize weights randomly ~N(0, $\sigma^{2}$)
* Loop until convergence:
* Compute gradient: $\dfrac{\partial J(W)}{\partial W}$
* Update weights, **W** <- **W** - $\eta\dfrac{\partial J(W)}{\partial W}$ ($\eta$ is the learning rate/step)
* Return weights
```python
import tensorflow as tf

weights = tf.Variable([tf.random.normal()])

while True: # Loop forever without convergence
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)

    # lr is learning rate, the size of the step
    weights = weights - lr * gradient
```
### Backpropogation
For this example we will calculate the gradient for some weight called $w_{2}$ with $\hat{y}$ as: the output
$\begin{equation}
\dfrac{\partial J(W)}{\partial w_{2}} = \dfrac{\partial J(W)}{\partial \hat{y}} * \dfrac{\partial \hat{y}}{\partial w_{2}}
\end{equation}$
$\begin{equation}
\dfrac{\partial J(W)}{\partial w_{1}} = \dfrac{\partial J(W)}{\partial \hat{y}} * \dfrac{\partial \hat{y}}{\partial z_{1}} * \dfrac{\partial z_{1}}{\partial w_{1}}
\end{equation}$

* Where $z_{1}$ is the gradient from $w_{1}$?
* Repeat this for every weight in the network using gradients from later layers
* We want the derviative of our loss with respect to every weight in the network

### Neural Networks in Practice: Optimization

* Backpropogation is very computationaly intensive
* Simple neural networks are 2D but real networks in practice are millon/billion dimensional so what does the loss landscape look like for these?
    + It is possible to landscape this but they are extremely messy
* Construct loss landscapes that are amenable to optimization
* Setting $\eta$ is difficult
    + If $\eta$ is too small it can get stuck and be very slow
    + If $\eta$ is too big it might overshoot and diverge from the local minimum
* How do we set $\eta$?
    + Could try a bunch of different $\eta$ and see what works
    + Could design an algorithm that can adapt to the landscape, $\eta$ doesnt need to be a single number
        - Even in tf there are many different optomization algorithms implemented

#### Putting it all together
```python
import tensorflow as tf

model = tf.keras.Sequential([...])

# Pick your favorite optimizer
optimizer = tf.keras.optimizer.SGD()

while True: # Loop forever

    # Forward pass throught the network
    prediction = model(x)

    with tf.GradientTape() as tape:
        # Compute the loss
        loss = compute_ loss(y, prediction)

    # Update the weights using the gradient
    grads = tape.Gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### Mini-batches
* Instead of an average of our entire data set, why not use batches
#### Stochastic Gradient Descent (SGD)
* Pick a single training point and compute its gradient
* Much cheaper computationally
#### Batch
* Instead of doing 1 or all, just do a batch (usually 32)
* Iterates much faster because its a much smaller set
* This leads to more accurate estimations of gradient
    + Smoother convergence
    + Allows for larger learning rates
* We can split up these batches to different parts of the GPU to calculate these in parallel so its much faster

### Overfitting
* We want our model to do well when exposed to new things, not just the training set data
* If your model is doing well with training data but no to new data its overfit
* Ideally you want something in the middle of underfitting and overfitting
#### Regularization
* Technique to discourage your model from learing the nuances in your training data
##### Dropout
* During training we'll set some activations to 0 with some probability
* So some amount of the neurons will be shutdown during training
* Next iteration pick a new set of neurons to shutoff so the network has to build new pathways and cannot rely on little details that it has learned from the training data
##### Early Stopping
* Model agnostic
* Look at performance on both training set and test set
    + Eventually the training loss plateus
    + We care about the moment when the training accuracy is still increasing but testing accuracy is getting worse