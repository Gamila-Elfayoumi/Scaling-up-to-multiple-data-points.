# Scaling-up-to-multiple-data-points.
This code compare model accuracies for two different sets of weights, which have been stored as weights_0 and weights_1.
•	Import mean_squared_error from sklearn.metrics.
•	Using a for loop to iterate over each row of input_data:
    o	Make predictions for each row with weights_0 using the predict_with_network() function and append it to model_output_0.
    o	Do the same for weights_1, appending the predictions to model_output_1.
•	Calculate the mean squared error of model_output_0 and then model_output_1 using the mean_squared_error() function. The first argument should be the actual values (target_actuals), and the second argument should be the predicted values (model_output_0 or model_output_1).
