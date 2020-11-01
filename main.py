import math
from perceptron import Perceptron
from fetch import DataSetType, FetchingService

def threshold_function(activation_point):
    return int(activation_point >= 0)

def sigmoid_function(activation_point):
    return 1 / (1 + math.exp(-activation_point))

def prediction_to_class(prediction, threshold):
    return int(prediction >= threshold)

if __name__ == '__main__':
    data_sets = DataSetType.SECOND_SET
    train_data, test_data = FetchingService.fetch_data(data_sets)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    activation_function = threshold_function
    threshold = 0.5 # only used if sigmoid function is provided
    max_iterations = 100
    learning_rate = 0.5
    
    perceptron = Perceptron(activation_function, max_iterations, learning_rate)
    perceptron.train(train_data, weights)
    print(perceptron.prediction_error)
    print(perceptron.weights)

    test_data_inputs = [group[:-1] for group in train_data]
    test_data_targets = [group[-1] for group in train_data]
    good_predictions = 0
    for input_group, target in zip(test_data_inputs, test_data_targets):
        prediction = perceptron.predict(input_group)
        good_predictions += prediction_to_class(prediction, threshold) == target
    print(good_predictions / len(test_data_targets))