#%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)


dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)


quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
	
	# last 21 days data
test_data = data[-21*24:]
data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / ( 1 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
         
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
      
        hidden_outputs = self.activation_function(hidden_inputs)
        
        
        #  Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
       
        final_outputs = final_inputs
        
        
        
        
        #Output error
        output_errors = targets - final_outputs
       
        
        # Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        
        
        hidden_grad =  hidden_outputs * (1 - hidden_outputs)
        
        
        # TUpdate the weights
        output_grad =1
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        
                                                                                                 
        self.weights_input_to_hidden += self.lr * np.dot( hidden_grad * hidden_errors, inputs.T)
     
                                                                                                 
 
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
                                                                                                 
        hidden_outputs = self.activation_function(hidden_inputs)
                                                                                                 
        
        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
                                                                                                
        final_outputs = final_inputs
                                                                                                  
        
        return final_outputs
		
def MSE(y, Y):
    return np.mean((y-Y)**2)
	
import sys


epochs = 2000
learning_rate = 0.075
hidden_nodes = 30
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
