import numpy as np


class Perceptron:

    def __init__(self, activation_function, compare_function=None, max_iterations=100, learning_rate=0.5):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weights = None
        self.prediction_error = 1

        if compare_function is not None:
            self.compare_function = compare_function
        else:
            self.compare_function = lambda prediction, target: prediction == target

    def __try_assign_weights_as_inputs(self, no_of_inputs, input_weights):
        BIAS_WEIGHT = 1
        if len(input_weights) != (no_of_inputs + BIAS_WEIGHT):
            err_str = "Weights array is either too big or too small. Weights array length: {0}, expected: {1}".format(
                len(input_weights), (no_of_inputs + BIAS_WEIGHT))
            raise ValueError(err_str)
        self.weights = np.array(input_weights)

    def __init_weights(self, no_of_inputs, input_weights=None):
        if input_weights is None:
            self.weights = np.zeros(no_of_inputs + 1)
        else:
            self.__try_assign_weights_as_inputs(no_of_inputs, input_weights)

    def __optimise_weights(self, error, input_group):
        self.weights[1:] += self.learning_rate * error * input_group
        self.weights[0] += self.learning_rate * error

    def __try_calculate_a(self, input_group):
        return np.dot(input_group, self.weights[1:]) + self.weights[0]

    def predict(self, input_group):
        a = self.__try_calculate_a(input_group)
        return self.activation_function(a)

    def __get_inputs_from_data(self, data):
        return [np.array(group[:-1]) for group in data]

    def __get_targets_from_data(self, data):
        return [group[-1] for group in data]

    def train(self, data, weights=None):
        inputs = self.__get_inputs_from_data(data)
        targets = self.__get_targets_from_data(data)
        self.__init_weights(len(inputs[0]), weights)

        for _ in range(self.max_iterations):
            good_predictions = 0
            is_model_edited = False
            for input_group, target in zip(inputs, targets):
                prediction = self.predict(input_group)
                error = abs(target - prediction)

                if not self.compare_function(prediction, target):
                    is_model_edited = True
                    self.__optimise_weights(error, input_group)
                else:
                    good_predictions += 1

            self.prediction_error = 1 - (good_predictions / len(inputs))

            if not is_model_edited:
                break
