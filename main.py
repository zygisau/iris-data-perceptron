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


def handle_significant_error(error, iteration_idx, target, prediction):
    if round(error[-1], 2) != 0:
        print(f"Prediction error occured on {iteration_idx} iteration; Target: {target}; prediction: {prediction}; "
              f"error: {round(error[-1], 2)}")


def add_to_output(output, inputs, prediction, target):
    output.append(f"Inputs: {inputs}, Predicted value: {round(prediction, 2)}, Expected value: {round(target, 2)}")


def output_to_file(output):
    file = open('predictions.txt', 'w')
    for line in output:
        file.write(line)
        file.write('\n')
    file.close()


def test_model(test_data):
    test_data_inputs = [group[:-1] for group in test_data]
    test_data_targets = [group[-1] for group in test_data]
    good_predictions = 0
    error = list()
    output = []
    for idx, (input_group, target) in enumerate(zip(test_data_inputs, test_data_targets)):
        prediction = perceptron.predict(input_group)
        error.append(abs(prediction - target))
        handle_significant_error(error, idx, target, round(prediction, 2))
        good_predictions += compare_target_to_prediction(prediction, target)
        add_to_output(output, input_group, prediction, target)

    error_rounded = [round(val, 2) for val in error]
    accuracy_percentage = "{0:.0%}".format(good_predictions / len(test_data_targets))
    output_to_file(output)

    print(f"Errors: {error_rounded}")
    print(f"Mean error: {round(mean(error), 2)}")
    print(f"Classification accuracy: {accuracy_percentage}")


if __name__ == '__main__':
    threshold = 0.5  # only used with sigmoid function

    data_sets = DataSetType.SECOND_SET
    train_data, test_data = FetchingService.fetch_data(data_sets)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    activation_function = threshold_function
    max_iterations = 100
    learning_rate = 0.8

    perceptron = Perceptron(activation_function=activation_function, compare_function=compare_target_to_prediction,
                            max_iterations=max_iterations, learning_rate=learning_rate, verbose=True)
    perceptron.train(train_data, weights)
    print(f"Weights: {[ '%.2f' % elem for elem in perceptron.weights ]}")

    test_model(test_data)
