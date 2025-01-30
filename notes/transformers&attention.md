# MIT 6.S191: Recurrent Neural Networks, Transformers, and Attention Notes
[Source](https://www.youtube.com/watch?v=dqoEU9Ac3ek&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2)
### Sequence Modeling
#### Sequence Modeling Applications
* Many to One
* One to Many
* Many to Many
#### Constructing a Model for Sequential Data
* If we map each input to an output at a given time, so $x_{t}$ maps to $\hat{y}_{t}$ then we have a series of networks that have not learned from each other. If we want our future predictions to learn from previous data then we need some connection
* Introduce a variable $h$ where $h_{t}$ is the link between the $x_{t}$ and $x_{t+1}$ networks
* Then we can determine $\hat{y}_{t} = f(x_{t}, h_{t-1})$ where $\hat{y}_{t}$ is the output, $x_{t}$ is the input and $h_{t-1}$ is the past memory.
### This architecture is a basic Recurrent Neural Network (RNN)
* We have a cell state variable $h_{t}$ that equals a function with weights W so $f_{W}$ with arguments $x_{t}$ and $h_{t-1}$ so we have $h_{t} = f_{W}(x_{t}, h_{t-1})$
* RNNs have a state $h_{t}$ that is updated at each time step as a sequence is processed
* We have the same function and set of parameters are used at every step
```python
my_rnn = RNN()
hidden_state = [0,0,0,0]

sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction
# >>> "networks!"
```
### RNNs from scratch
```python
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        # Intialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        # Intialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])
    
    def call(self, x):
        # Update the hidden state
        self.h = tf.math.tanh( self.W_hh * self.h + self.W_xh * x)

        # Compute the output
        output = self.W_hy * self.h

        # Return the current output and hidden state
        return output, self.h
```
#### Sequence Modeling: Design Criteria
1. Handle **variable-length** sequences
1. Track **long-term** dependencies
1. Maintain information about **order**
1. **Share parameters** across the sequence
RNNS meet these criteria
### Sequence Modeling: predict the next word
"This morning I took my cat for a walk"
* given "This morning I took my cat for a" predict "walk"
* We need a way to represent a word numerically so that we can represent it in our math
* Our network needs to be able to handle different sized sequences or sentences
#### Embedding: transform indexes into a vector of a fixed size
* We could just map each word to a distinct index then create an array with the length of every word in our vocabulary and indicate each word but putting a 1 in the arrays index matching that word
* We could learn to map words to a space where things that are similar are close to each other numerically and things that are disimilar are far away from each other
#### Backpropogation through time (BPTT)
* In a standard FFNN we take the derivative of the loss with respect to each parameter, then shift parameters in order to minimize loss
* With RNNS, theres a wrinkle because we have a loss that is computed time step to timestep
    + We now have to backpropogate the gradients per each timestep then across all time steps from the end going toward the beginning
* Gradients can explode to very large numbers or to very small numbers but there are techniques to help with this
* Why are vanishing gradients a problem?
    + You lose track of being able to learn something useful from previous time steps
        - We can pick smart activation function such as the ReLU function
        - we can initialize the weights smartly from the start to reduce the liklihood of this happening in the future
        - Gated Cells: introduce additional computation in each recurrent cell to selectively add/remove information (this makes long short term memory networks possible)
#### LSTMS
1. maintain a **cell-state**
1. Use **gates** to control the **flow of information**
* **Forget** gate gets rid of irrelevant information
* **Store** relevant information from current input
* Selectively **update** cell state
* **Output** gate returns a filtered version of the celll state
3. Backpropogation through time with partially **uninterupted gradient flow**
### RNN Application
* Example Task:Music Generation
    + input: sheet music
    + output: next character in sheet music
* Example Task: Sentiment Classification
    + input: sequence of words
    + output: probability of having positive sentiment
#### Limitations of RNNs
* Encoding bottleneck - Information is long sequences can be lost by imposing a bottleneck
* Slow, no parallelization
* Not long memory
#### Goal of sequence modeling
* We want to take a series of outputs to learn some features then generate some predictive outputs
* Desired capabilities
    + Continous Stream
    + parallelization
    + Long memory
* What if we eliminated a need to eliminate our models need to handle data sequentially
    + We eliminate recurrence, but
    + not scalable
    + no order
    + no long memory
* Instead of thinking of things time step by time step, we learn a model that can tell us what parts of the sequence are the important parts that we should actually care about
### Attention is all you need
* The transformer architecture is built on attention being the most important concept
* Attending to the most important of the input
    + Identify which parts to attend to
    + Extract the features with high attention
* Understanding attention with search
    + Start with a query Q, then match Q with other Keys $K_{t}$ and determine how similar they are to find the relevant key
    + Extract values based on attention: return the values of highest attention
* Goal: identify and attend to most important features of an input
1. Encode **position** information
    * Achieve a position aware embedding
1. Extract **query,key,value** for search
1. Compute **attention weighting**
    * **Attention Score**: compute pairwise similarity between each query and key
        + $\dfrac{Q \cdot K^{t}}{scaling}$ - similarity metric or cosine similarity
    * Attention weighting: what is the relevant words that are similar to the key
    * We then squish this score to be between 0-1 to have a real weight
1. Extract features with high attention
    * $softmax(\dfrac{Q \cdot K^{t}}{scaling}) \cdot V$\

All of these put together form a single self-attention head and cteate a hierachy of these to create a transformer architecure.
* We can put these heads in parallel and stack them together to attend to different details and features
