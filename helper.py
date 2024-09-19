import random as rand


def random():
    return rand.gauss(0, 1.5)


def display(arr):
    for i in range(len(arr)):
        print("\t[", end="")
        for j in range(len(arr[0])):
            print(arr[i][j], end=" ")
        print("]")


def dot(matrix1, matrix2):
    product_matrix = []
    for i, row in enumerate(matrix1):
        product_row = []
        for j in range(len(matrix2[0])):
            col = [matrix2[r][j] for r in range(len(matrix2))]
            product_row.append(sum(a*b for a, b in zip(row, col)))
        product_matrix.append(product_row)
    return product_matrix






A = [[3, -1, 1],
     [-15, 6, -5],
     [5, -2, 2]]

B = [[3, 1, 2],
     [2, 1, 2],
     [6, 2, 5]]



display(dot(A,B))



def const_matrix(constant, size=(0, 0), sample_copy=[[]]):
    if size != (0, 0):
        rows, cols = size
    else:
        rows, cols = len(sample_copy), len(sample_copy[0])
    return [[constant for _ in range(cols)]for _ in range(rows)]


def add(matrix1, matrix2):
    sum_matrix = []
    for i in range(len(matrix1)):
        sum_row = []
        for j in range(len(matrix1[0])):
            sum_row.append(matrix1[i][j] + matrix2[i][j])
        sum_matrix.append(sum_row)

    return sum_matrix


def multiply(matrix1, matrix2):
    multiply_matrix = []
    for i in range(len(matrix1)):
        sum_row = []
        for j in range(len(matrix1[0])):
            sum_row.append(matrix1[i][j] * matrix2[i][j])
        multiply_matrix.append(sum_row)

    return multiply_matrix


def transpose(matrix):
    transposed_matrix = []
    for j in range(len(matrix[0])):
        col = [matrix[r][j] for r in range(len(matrix))]
        transposed_matrix.append(col)
    return transposed_matrix


def mean(matrix):
    # only implented for column matrix
    summ = sum([matrix[r][0] for r in range(len(matrix))])
    mean = summ / len(matrix)
    return mean


def derivative(func, x):
    h = 0.00000001
    ans = (func(x+h) - func(x))/h
    return ans


def apply_function(matrix, func):
    applied_matrix = []
    for row in matrix:
        row_matrix = []
        for item in row:
            row_matrix.append(func(item))
        applied_matrix.append(row_matrix)
    return applied_matrix


def max_matrix(matrix):
    return max(matrix)[0]
