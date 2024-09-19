import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from Dense import Dense
from Activation_func import TanH, ReLU, Sigmoid, Step, Softmax
from error import mse, mse_derivative
import pandas as pd

X = [
    [[1], [1]],
    [[1], [0]],
    [[0], [1]],
    [[0], [0]],
]

Y = [
    [0],
    [1],
    [1],
    [0]
]

data = pd.read_csv("BMI_data.csv")
Weights = list(data["Weight"])[:100]
Heights = list(data["Height"])[:100]

# training_range = round(len(Weights) * 0.8)
# X = list([[[w], [h]]
#          for w, h in zip(Weights[:training_range], Heights[:training_range])])
# Y = [[item] for item in list(data["Index"])[:training_range]]

network = [
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
    TanH()
]

epochs = 200_000
learning_rate = 0.1

for e in range(epochs):
    error = 0

    # forward
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            # print(output)
            output = layer.forward(output)

        error += mse([y], output)

        # backward
        grad = mse_derivative([y], output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print(f"{e+1}/{epochs}. error={error}")


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def plot_xor():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = predict(network, [[x], [y]])
            points.append([x, y, z[0]])

    X = [points[r][0] for r in range(len(points))]
    Y = [points[r][1] for r in range(len(points))]
    Z = [points[r][2] for r in range(len(points))]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


plot_xor()
# X_test = list([[[w], [h]]
#                for w, h in zip(Weights[training_range:], Heights[training_range:])])
# Y_test = [[item] for item in list(data["Index"])[training_range:]]

# acc = 0
# for i, x in enumerate(X_test):
#     y = predict(network, x)
#     if y == Y_test[i]:
#         acc += 1
#     print(y, Y_test[i])
# print(acc)
