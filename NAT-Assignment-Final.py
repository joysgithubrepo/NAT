#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Import PySwarms

import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)


# In[ ]:


# Load the data-frame
# Add two more additional columns for sin functions

ds = pd.read_csv("two_spirals.dat")
ds["SINX1"] = np.sin(ds["X1"])
ds["SINX2"] = np.sin(ds["X2"])

# Split the dataset in train test
X_train, X_test, y_train, y_test = train_test_split(ds[["X1", "X2", "SINX1", "SINX2"]], ds["Y"], test_size=0.50)
#X_train, X_test, y_train, y_test = train_test_split(ds[["X1", "X2"]], ds["Y"], test_size=0.50)


# In[ ]:


# Store the features as X and the labels as y

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()


# In[ ]:


# Set the neural network architecture

n_inputs = 4
n_hidden1 = 6
n_hidden2 = 6
n_classes = 2
num_samples = len(X_train)

# Set the swarm parameters

dimensions = n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2 + n_hidden2*n_classes + n_classes

n_particles = 100
n_iters = 2000
# d_options = {'c1': 0.5, 'c2': 0.9, 'w': 0.9}
d_options = {'c1': 0.5, 'c2': 0.6, 'w': 0.9}

# Create bounds
# max_bound = 5.12 * np.ones(2)
# min_bound = - max_bound
# bounds = (min_bound, max_bound)


# In[ ]:


# Define the function

def logits_function(params, X):

    """ Calculate roll-back the weights and biases

    Inputs
    ------
    p: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    numpy.ndarray of logits for layer 3

    """
    # Roll-back the weights and biases
    W1 = params[0:n_inputs*n_hidden1].reshape((n_inputs,n_hidden1))
    b1 = params[n_inputs*n_hidden1:n_inputs*n_hidden1 + n_hidden1].reshape((n_hidden1,))
    W2 = params[n_inputs*n_hidden1 + n_hidden1:n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2].reshape((n_hidden1,n_hidden2))
    b2 = params[n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2:n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2].reshape((n_hidden2,))
    W3 = params[n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2:n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2 + n_hidden2*n_classes].reshape((n_hidden2,n_classes))
    b3 = params[n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2 + n_hidden2*n_classes:n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2 + n_hidden2*n_classes + n_classes].reshape((n_classes,))   

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    a2 = np.tanh(z2)     # Activation in Layer 2
    z3 = a2.dot(W3) + b3 # Pre-activation in Layer 3
    logits = z3          # Logits for Layer 3
    
    return logits


# In[ ]:


# Forward propagation

def forward_prop(params):

    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """

    logits = logits_function(params, X_train)

    # Compute for the softmax of the logits

    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    corect_logprobs = -np.log(probs[range(num_samples), y_train])
    loss = np.sum(corect_logprobs) / num_samples

    return loss


# In[ ]:


# Define the function to forward propagate through the entire swarm population

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)


# In[ ]:


# Call instance of PSO
# dimensions = n_inputs*n_hidden1 + n_hidden1 + n_hidden1*n_hidden2 + n_hidden2 + n_hidden2*n_classes + n_classes
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=d_options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=n_iters)


# In[ ]:


plot_cost_history(cost_history=optimizer.cost_history)
plt.show()


# In[ ]:


def predict(pos, X):

    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    logits = logits_function(pos, X)
    y_pred = np.argmax(logits, axis=1)
    return y_pred


# In[ ]:


y_pred = predict(pos, X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


y_test


# In[ ]:


y_pred


# In[ ]:




