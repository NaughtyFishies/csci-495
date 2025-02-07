import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))

class Neuron:
    def __init__(self, next_layer_size):
        self.weights: list = []
        for _ in range(next_layer_size):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

    def output(self, inputs):
        return sigmoid(dot_product(self.weights, inputs) + self.bias)


class NeuralNetwork:
    def __init__(self, number_of_inputs: int, hidden_layers: list[int], number_of_outputs: int):
        self.input: list[Neuron] = []
        self.output: list[Neuron] = []
        self.hidden: list[list[Neuron]] = []
        
        if hidden_layers == []:
            next_layer_size = number_of_outputs
        else:
            next_layer_size = hidden_layers[0]

        # Initializing the input layer
        for _ in range(number_of_inputs):
            self.input.append(Neuron(next_layer_size))
        
        # Intializing the output layer
        for _ in range(number_of_outputs):
            self.output.append(Neuron(0))
        
        # Intializing the hidden layers
        for i in range(len(hidden_layers)):
            layer: list[Neuron] = []
            if i == len(hidden_layers) - 1:
                next_layer_size = number_of_outputs
            else:
                next_layer_size = hidden_layers[i + 1]

            for _ in range(hidden_layers[i]):
                layer.append(Neuron(next_layer_size))
            self.hidden.append(layer)

        self.learning_rate = 0.1


    def feedforward(self, inputs):
        
        # Process input layer
        layer_outputs = [neuron.output(inputs) for neuron in self.input]
        
        # Process hidden layers
        for layer in self.hidden:
            layer_outputs = [neuron.output(layer_outputs) for neuron in layer]
        
        # Process output layer
        output_values = [neuron.output(layer_outputs) for neuron in self.output]
        
        return output_values


def main():
    nn = NeuralNetwork(3, [4, 5], 2)  # 3 inputs, 2 hidden layers (4 & 5 neurons), 2 outputs
    input_data = [0.5, 0.8, -0.3]
    output = nn.feedforward(input_data)
    print(output)

if __name__ == "__main__":
    main()