#!/usr/bin/env python

import numpy as np
from scipy.special import expit # sigmoid function :: ( 1 / (1 + np.exp(-X)) )

def derivative_sigmoid(X):
    sig = expit(X)
    return sig * (1 - sig)


class MLP(object):
    """ MULTILAYER PERCEPTRON with 1 hidden layer """

    # Step 0: Read input and output
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # Step 1: Initialize weights and biases with random values
    def Initialize_Weights_Bias(self):
        self.wh = np.random.rand(4, 3) # Random Weight for hidden layer
        self.wh = np.array([[0.42, 0.88, 0.55], [0.10, 0.73, 0.68], [0.60, 0.18, 0.47], [0.92, 0.11, 0.52]]) # Use values in Session 2
        self.bh = np.random.rand(1, 3) # Random Bias for hidden layer
        self.bh = np.array([0.46, 0.72, 0.08]) # Use values in Session 2

        self.wout = np.random.rand(3, 1) # Random Weight for output layer
        self.wout = np.array([[0.30], [0.25], [0.23]]) # Use values in Session 2
        self.bout = np.random.rand(1) # Random Bias for output layer
        self.bout = np.array([0.69]) # Use values in Session 2


    # Step 2: Calculate hidden layer input
    def calculate_hidden_layer_input(self):
        self.hidden_layer_input = np.dot(self.X, self.wh) + self.bh
        print "hidden layer input"
        print self.hidden_layer_input
        print

    # Step 3: Perform non-linear transformation on hidden linear input
    def activation_function(self):
        self.hiddenlayer_activations = expit(self.hidden_layer_input)
        print "hiddenlayer_activations"
        print self.hiddenlayer_activations # Final output from hidden layer
        print

    # Step 4: Output Layer Input + Activation function
    def output_layer(self):
        self.output_layer_input = np.dot(self.hiddenlayer_activations, self.wout) + self.bout
        print self.output_layer_input 
        self.output = expit(self.output_layer_input)
        print "\nOutput at output layer"
        print self.output # Final output from output layer
        print

    # Step 5: Error at output layer
    def error(self):
        self.E = Y - self.output
        print "Output Layer Error"
        print self.E
        print

    # Step 6: Compute slope at output and hidden layer
    def slope(self):
        self.slope_output_layer = derivative_sigmoid(self.output)
        print "slope output"
        print self.slope_output_layer

        self.slope_hidden_layer = derivative_sigmoid(self.hiddenlayer_activations)
        print "\nslope hidden layer"
        print self.slope_hidden_layer

    # Step 7: Compute delta at output layer
    def delta_output_layer(self):
        self.learning_rate = 0.1
        self.d_output = self.E * self.slope_output_layer * self.learning_rate
        print "delta_output_layer ", self.d_output.shape
        print self.d_output

    # Step 8: Calculate Error at hidden layer
    def error_at_hidden_layer(self):
        self.error_hidden_layer = np.dot(self.d_output, self.wout.T)
        print "error_hidden_layer"
        print self.error_hidden_layer

    # Step 9: Compute delta at hidden layer
    def delta_hidden_layer(self):
        self.d_hiddenlayer = self.error_hidden_layer * self.slope_hidden_layer
        print "\nd_hiddenlayer"
        print self.d_hiddenlayer

    # Step 10: Update weight at both output and hidden layer
    def update_weights(self):
        self.wout = self.wout + np.dot(self.hiddenlayer_activations.T, self.d_output) * self.learning_rate
        print "\nwout"
        print self.wout
        self.wh = self.wh + np.dot(self.X.T, self.d_hiddenlayer) * self.learning_rate
        print "\nwh"
        print self.wh

    # Step 11: Update biases at both output and hidden layer
    def update_biases(self):
        self.bh = self.bh + np.sum(self.d_hiddenlayer, axis=0) * self.learning_rate
        print "\nbh"
        print self.bh
        self.bout = self.bout + np.sum(self.d_output, axis=0) * self.learning_rate
        print "\nbout"
        print self.bout


if __name__ == '__main__':
    X = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]]) # Input 3X4
    Y = np.array([[1], [1], [0]]) # Expected Output 3X1
    
    model = MLP(X, Y) # Step 0: Read input and output
    model.Initialize_Weights_Bias() # Step 1: Initialize weights and biases with random values
    model.calculate_hidden_layer_input() # Step 2: Calculate hidden layer input
    model.activation_function() # Step 3: Perform non-linear transformation on hidden linear input
    model.output_layer() # Step 4: Output Layer Input + Activation function
    model.error() # Error at output layer
    model.slope() # Step 6: Compute slope at output and hidden layer
    model.delta_output_layer() # Step 7: Compute delta at output layer
    model.error_at_hidden_layer() # Step 8: Calculate Error at hidden layer
    model.delta_hidden_layer() # Step 9: Compute delta at hidden layer
    model.update_weights() # Step 10: Update weight at both output and hidden layer
    model.update_biases() # Step 11: Update biases at both output and hidden layer

    





