import numpy as np
from sklearn.metrics import mean_squared_error
def predict_with_network(input_data, weights):
    for layer_name, layer_weights in weights.items():
        input_data = np.dot(input_data, layer_weights)
    return input_data
# Define the input data
input_data = np.array([[0, 3], [1, 2], [-1, -2], [4, 0]])

# Define the target actuals
target_actuals = np.array([1, 3, 5, 7])

# Define the weights for the first model
weights_0 = {'node_0': np.array([2, 1]), 'node_1': np.array([1, 2]), 'output': np.array([1, 1])}

# Define the weights for the second model
weights_1 = {'node_0': np.array([2, 1]), 'node_1': np.array([1., 1.5]), 'output': np.array([1., 1.5])}

# Create the model outputs for the first model
model_output_0 = []
for row in input_data:
    model_output_0.append(predict_with_network(row, weights_0))

# Create the model outputs for the second model
model_output_1 = []
for row in input_data:
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for the first model
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for the second model
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print the mean squared errors
print("Mean squared error with weights_0: %f" % mse_0)
print("Mean squared error with weights_1: %f" % mse_1)
