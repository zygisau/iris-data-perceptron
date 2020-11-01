import numpy as np


class Perceptron:

    def __init__(self, activation_function, compare_function=None, max_iterations=100,
                 learning_rate=0.5, verbose=False):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weights = None
        self.prediction_error = 1
        self.verbose = verbose

        if compare_function is not None:
            self.is_prediction_equal_to_target_func = compare_function
        else:
            self.is_prediction_equal_to_target_func = lambda prediction, target: prediction == target

        self.__print_message_if_verbose(f"Perceptron has been created. Settings: \n"
                                        f"learning rate: {self.learning_rate} \n"
                                        f"max_iterations: {self.max_iterations} \n")

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

    def __print_message_if_verbose(self, msg):
        if self.verbose:
            print(msg)

    def train(self, data, weights=None):
        inputs = self.__get_inputs_from_data(data)
        targets = self.__get_targets_from_data(data)
        self.__init_weights(len(inputs[0]), weights)

        for epoch in range(self.max_iterations):
            self.__print_message_if_verbose(f"Epoch: {epoch + 1} starting...")

            good_predictions = 0
            is_model_edited = False
            for input_group, target in zip(inputs, targets):
                prediction = self.predict(input_group)
                error = target - prediction

                if not self.is_prediction_equal_to_target_func(prediction, target):
                    is_model_edited = True
                    self.__optimise_weights(error, input_group)
                else:
                    good_predictions += 1

            self.prediction_error = 1 - (good_predictions / len(inputs))

            self.__print_message_if_verbose(f"Epoch has ended with: good predictions: {good_predictions}"
                                            f" / {len(inputs)}; prediction error: {self.prediction_error}; "
                                            f"weights {self.weights}\n")

            if not is_model_edited:
                self.__print_message_if_verbose(f"Model passed training earlier than max iterations\n")
                break

        self.__print_message_if_verbose(f"Model is trained\n")
