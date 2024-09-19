from helper import *


def mse(Y_true, Y_pred):
    neg_matrix = const_matrix(-1, sample_copy=Y_pred)
    neg_Y_pred = multiply(Y_pred, neg_matrix)

    true_minus_pred = add(Y_true, neg_Y_pred)
    true_minus_pred_squared = multiply(true_minus_pred, true_minus_pred)
    return mean(true_minus_pred_squared)


def mse_derivative(Y_true, Y_pred):
    neg_matrix = const_matrix(-1, sample_copy=Y_true)
    neg_Y_true = multiply(Y_true, neg_matrix)
    pred_minus_true = add(Y_pred, neg_Y_true)

    constant = 2/len(Y_true)
    constant_matrix = const_matrix(constant, sample_copy=pred_minus_true)

    error_matrix = multiply(pred_minus_true, constant_matrix)
    return error_matrix
