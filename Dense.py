from helper import *
from Layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = [[random() for _ in range(self.input_size)]
                        for _ in range(self.output_size)]
        self.biases = transpose([[random() for _ in range(self.output_size)]])

    def forward(self, layer_input):
        self.layer_input = layer_input
        input_weight_product = dot(self.weights, self.layer_input)
        output = add(input_weight_product, self.biases)
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = dot(output_gradient, transpose(self.layer_input))
        input_gradient = dot(transpose(self.weights), output_gradient)

        learning_rate_matrix_weights = const_matrix(
            -1*learning_rate, sample_copy=self.weights)
        weight_correction_matrix = multiply(
            weights_gradient, learning_rate_matrix_weights)

        learning_rate_matrix_bias = const_matrix(
            -1*learning_rate, sample_copy=self.biases)
        bias_correction_matrix = multiply(
            output_gradient, learning_rate_matrix_bias)

        self.weights = add(self.weights, weight_correction_matrix)
        self.biases = add(self.biases, bias_correction_matrix)
        return input_gradient
