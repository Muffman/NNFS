from Activation import Activation
from helper import derivative, apply_function, max_matrix, const_matrix, add
e = 2.718281828459045


class TanH(Activation):
    def __init__(self):
        super().__init__(lambda arr: self.tanH(arr), lambda arr: self.tanH_derivative(arr))

    def tanH_func(self, x):
        return ((e**x) - (e**-x)) / \
            ((e**x) + (e**-x))

    def tanH(self, matrix):
        return apply_function(matrix, self.tanH_func)

    def tanH_derivative(self, matrix):
        return apply_function(matrix, lambda x: derivative(self.tanH_func, x))


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda arr: self.relu(arr), lambda arr: self.relu_derivative(arr))

    def relu_func(self, x):
        ans = x if x > 0 else 0
        return ans

    def relu(self, matrix):
        return apply_function(matrix, self.relu_func)

    def relu_derivative(self, matrix):
        return apply_function(matrix, lambda x: derivative(self.relu_func, x))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(lambda arr: self.sigmoid(arr),
                         lambda arr: self.sigmoid_derivative(arr))

    def sigmoid_func(self, x):
        return ((1)/(1+(e**-x)))

    def sigmoid(self, matrix):
        return apply_function(matrix, self.sigmoid_func)

    def sigmoid_derivative(self, matrix):
        return apply_function(matrix, lambda x: derivative(self.sigmoid_func, x))


class Step(Activation):
    def __init__(self):
        super().__init__(lambda arr: self.step(arr),
                         lambda arr: self.step_derivative(arr))

    def step_func(self, x):
        ans = 1 if x > 0 else 0
        return ans

    def step(self, matrix):
        return apply_function(matrix, self.step_func)

    def step_derivative(self, matrix):
        return apply_function(matrix, lambda x: derivative(self.step_func, x))


class Softmax(Activation):
    def __init__(self):
        super().__init__(lambda arr: self.softmax(arr),
                         lambda arr: self.softmax_derivative(arr))

    def softmax_func(self, x):
        ans = e**x
        return ans

    def softmax(self, matrix):
        matrix = self.minimize_values(matrix)
        return apply_function(matrix, self.softmax_func)

    def softmax_derivative(self, matrix):
        matrix = self.minimize_values(matrix)
        return apply_function(matrix, lambda x: derivative(self.softmax_func, x))

    def minimize_values(self, matrix):
        maximum = max_matrix(matrix)
        constant_matrix = const_matrix(-maximum, sample_copy=matrix)
        minimized_matrix = add(matrix, constant_matrix)
        return minimized_matrix


