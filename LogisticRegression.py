import h5py
import numpy as np
import math
import random


def sigmoid(a):
    if a > 100:
        return 1
    elif a < -100:
        return 0.001
    else:
        return 1 / (1 + math.exp(-a))


def calcYhat(theta, phi):
    # theta is the basis vector
    # phi is the vector of input data with a 1 appended to the beginning
    activation = np.dot(theta.T, phi)
    yhat = sigmoid(activation)
    return yhat


def gradient(theta, phi, t, sigma, m):
    # Theta is the basis vectors, a vector of coefficients for each feature value
    # phi is the vector of input data with a 1 appended to the beginning
    # t is a vector of labels from training data
    # sigma is the covariance of the prior distribution
    # m is the mean of the prior distribution
    sigma_inv = np.linalg.inv(sigma)
    basis_term = np.matmul(sigma_inv, theta - m)
    regression_term = 0
    for n in range(len(theta)):
        yhat = calcYhat(theta, phi[n])
        regression_term += (yhat - t[n]) * phi[n]

    return basis_term + regression_term


def objective(theta, phi, t, sigma, m):
    # Theta is the basis vectors, a vector of coefficients for each feature value
    # phi is the vector of input data with a 1 appended to the beginning
    # t is a vector of labels from training data
    # sigma is the covariance of the prior distribution
    # m is the mean of the prior distribution

    prior_exp = 0.5 * len(phi[0]) * math.log(2 * math.pi) + 0.5 * math.log(np.linalg.norm(sigma))
    a = np.matmul((theta - m).T, np.linalg.inv(sigma))
    b = np.matmul(a, (theta - m).T)
    prior_exp += 0.5 * b
    bernulli_exp = 0
    for n in range(len(t)):
        yhat = calcYhat(theta, phi[n])
        bernulli_exp += t[n] * np.log(yhat) + (1 - t[n]) * (1 - yhat)
    obj = prior_exp - bernulli_exp
    return obj


def descent(theta, phi, t, sigma, m, eta=10, epsilon=0.01):
    # Theta is the basis vectors, a vector of coefficients for each feature value
    # phi is the vector of input data with a 1 appended to the beginning
    # t is a vector of labels from training data
    # sigma is the covariance of the prior distribution
    # m is the mean of the prior distribution
    # eta is the step size for the gradient descent
    # epsilon is the threshold that stops the descent when eta is less than or equal to epsilon

    eta_start = eta
    while eta > epsilon:
        grad_of_obj = gradient(theta, phi, t, sigma, m)
        theta_updated = theta - eta * grad_of_obj
        o_u = objective(theta_updated, phi, t, sigma, m)
        o_c = objective(theta, phi, t, sigma, m)
        print("objective of updated theta", o_u)
        print("objective of current theta", o_c)
        print("eta", eta)
        if o_u < o_c:
            eta *= 2
            theta = theta_updated
        else:
            eta /= 2
    return theta


def logisticRegression(train_f, train_l, test_f, test_l):
    # Inputs are training and test data

    # Initialize phi matrix
    phi_ones = np.zeros((1, len(train_f[0]))) + 1
    phi = np.vstack((phi_ones, train_f)).T

    # Initialize theta vector
    theta = np.array([random.random() for _ in range(len(phi[0]))]) * 1000

    # Reshape t vector of labels
    t = np.reshape(train_l, (len(train_l[0])))

    # Initialize sigma, covariance of prior
    sigma = 2 * np.identity(len(phi[0]))

    # Initialize mean of prior
    m = np.zeros(len(phi[0]))

    # Perform gradient descent
    theta_optimized = descent(theta, phi, t, sigma, m)

    # Compute predictions from theta_optimized
    logistic_output = []
    for phi_n in phi:
        a = np.dot(theta_optimized.T, phi_n)
        logistic_output.append(sigmoid(a))
    preds = np.array([1 if (l > 0.5) else 0 for l in logistic_output])

    # Calculate Error between predictions and train t labels
    error = sum(1 for x, y in zip(preds, t) if x != y) / len(preds)
    print("Error rate over training data", error)

    # Test model with test data
    test_phi_ones = np.zeros((1, len(test_f[0]))) + 1
    test_phi = np.vstack((test_f, test_phi_ones)).T

    # Compute predictions from theta_optimized
    test_logistic_output = []
    for phi_n in test_phi:
        a = np.dot(theta_optimized.T, phi_n)
        test_logistic_output.append(sigmoid(a))
    preds = np.array([1 if (l > 0.5) else 0 for l in test_logistic_output])

    # Reshape t vector of labels
    t = np.reshape(test_l, (len(test_l[0])))
    # Calculate Error between predictions and test t labels
    test_error = sum(1 for x, y in zip(preds, t) if x != y) / len(preds)
    print("Error rate over testing data", test_error)


if __name__ == '__main__':
    f = h5py.File('/Users/dominickperini/Documents/2020_Fall/Machine_Learning/Assignment6/mnist01.h5', 'r')

    train_data = f["train"]
    test_data = f["test"]

    tr_feat = f["train/features"][()]
    tr_lab = f["train/labels"][()]
    te_feat = f["test/features"][()]
    te_lab = f["test/labels"][()]
    logisticRegression(tr_feat, tr_lab, te_feat, te_lab)
