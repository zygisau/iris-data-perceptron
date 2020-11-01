import math
from perceptron import Perceptron
from fetch import DataSetType, FetchingService


def threshold_function(activation_point):
    return int(activation_point >= 0)


def sigmoid_function(activation_point):
    return 1 / (1 + math.exp(-activation_point))


def compare_target_to_prediction(prediction_value, target_value):
    prediction_class = int(prediction_value >= threshold)
    return prediction_class == target_value


if __name__ == '__main__':
    threshold = 0.5  # only used with sigmoid function

    data_sets = DataSetType.FIRST_SET
    train_data, test_data = FetchingService.fetch_data(data_sets)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    activation_function = sigmoid_function
    max_iterations = 100
    learning_rate = 0.5

    perceptron = Perceptron(activation_function=activation_function, compare_function=compare_target_to_prediction,
                            max_iterations=max_iterations, learning_rate=learning_rate, verbose=False)
    perceptron.train(train_data, weights)
    print(perceptron.prediction_error)
    print(perceptron.weights)

    test_data_inputs = [group[:-1] for group in train_data]
    test_data_targets = [group[-1] for group in train_data]
    good_predictions = 0
    for input_group, target in zip(test_data_inputs, test_data_targets):
        prediction = perceptron.predict(input_group)
        good_predictions += compare_target_to_prediction(prediction, target)
    print(good_predictions / len(test_data_targets))
