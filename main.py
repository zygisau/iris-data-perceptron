import math
from statistics import mean

from perceptron import Perceptron
from fetch import DataSetType, FetchingService


def threshold_function(activation_point):
    return int(activation_point >= 0)


def sigmoid_function(activation_point):
    return 1 / (1 + math.exp(-activation_point))


def compare_target_to_prediction(prediction_value, target_value):
    prediction_class = int(prediction_value >= threshold)
    return prediction_class == target_value


def test_model(test_data):
    test_data_inputs = [group[:-1] for group in test_data]
    test_data_targets = [group[-1] for group in test_data]
    good_predictions = 0
    error = list()
    for input_group, target in zip(test_data_inputs, test_data_targets):
        prediction = perceptron.predict(input_group)
        error.append(abs(prediction - target))
        good_predictions += compare_target_to_prediction(prediction, target)

    error_percentage = "{0:.0%}".format(mean(error))
    error_rounded = [round(val, 2) for val in error]
    print(f"Errors: {error_rounded}")
    print(f"Mean error: {error_percentage}")

    accuracy_percentage = "{0:.0%}".format(good_predictions / len(test_data_targets))
    print(f"Classification accuracy: {accuracy_percentage}")


if __name__ == '__main__':
    threshold = 0.5  # only used with sigmoid function

    data_sets = DataSetType.FIRST_SET
    train_data, test_data = FetchingService.fetch_data(data_sets)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    activation_function = threshold_function
    max_iterations = 100
    learning_rate = 0.5

    perceptron = Perceptron(activation_function=activation_function, compare_function=compare_target_to_prediction,
                            max_iterations=max_iterations, learning_rate=learning_rate, verbose=False)
    perceptron.train(train_data, weights)
    print(f"Weights: {[ '%.2f' % elem for elem in perceptron.weights ]}")

    test_model(test_data)
