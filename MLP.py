# Kenzie Gage, using a multi-layer perceptron (MLP) to classify hand written numbers
# The datasets I used can be found at https://www.kaggle.com/oddrationale/mnist-in-csv
import numpy as np
from sklearn.metrics import confusion_matrix
import csv
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, training_file, testing_file, learning_rate, num_hidden, momentum):
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden
        self.momentum = momentum

        # Read in training data
        training_data = np.loadtxt(fname=training_file, dtype=str, delimiter=",", skiprows=1)
        training_input = np.asarray(training_data[0:, 1:], dtype=float)
        training_output = np.asarray(training_data[0:, 0:1], dtype=float)

        # Read in testing data
        testing_data = np.loadtxt(testing_file, dtype=str, delimiter=",", skiprows=1)
        testing_input = np.asarray(testing_data[0:, 1:], dtype=float)
        testing_output = np.asarray(testing_data[0:, 0:1], dtype=float)

        # Normalize data
        training_input = training_input / 255
        testing_input = testing_input / 255

        # adds a column of 1's to the beginning of a dataset
        def appendOnes(input_set):
            r, c = np.shape(input_set)
            ones_column = np.ones((r, 1))
            return np.hstack((ones_column, input_set))

        # add the bias
        training_input = appendOnes(training_input)
        testing_input = appendOnes(testing_input)

        self.testing_input = testing_input
        self.testing_output = testing_output

        # # Not splitting data up
        self.training_input = training_input
        self.training_output = training_output

        # Using half the data
        # self.training_input, a, self.training_output, b = train_test_split(
        #     self.training_input, self.training_output, test_size=30000, random_state=42)

        # Using 1/4 the data
        # self.training_input, a, self.training_output, b = train_test_split(
        #     self.training_input, self.training_output, test_size=15000, random_state=42)

        # Create an array of random weights between -.05 and .05
        def createWeights(rows, columns):
            weights = np.random.rand(rows, columns) * .1 - .05
            return weights

        # Set up two sets of weights
        r, c = np.shape(self.training_input)
        self.weights1 = createWeights(rows=c, columns=self.num_hidden)
        self.weights2 = createWeights(rows=(num_hidden + 1), columns=10)

    # Calculates the change in weights either between the input and hidden or hidden and output
    def changeInWeight(self, layer, delta, previous):
        # print(np.shape(layer), np.shape(delta), np.shape(previous))
        change = self.learning_rate * (np.dot(np.transpose(layer), delta)) + self.momentum * previous
        return change

    # Squash activations
    def sigmoidFunction(self, activations):
        r, c = np.shape(activations)
        squashed = np.ones((r, c))
        for i in range(0, c):
            squashed[0][i] = 1/(1 + np.exp(-1 * activations[0][i]))
        return squashed

    def doEpoch(self, input_set, output_set, perform_updates):
        r, c = np.shape(input_set)
        correct = 0
        total = 0

        # Need to keep track of the previous delta weights to incorporate momentum
        previous_change_w1 = np.zeros((np.shape(self.weights1)))
        previous_change_w2 = np.zeros((np.shape(self.weights2)))

        # Use these to create the confusion matrix for test data
        predicted_list = []
        actual_list = []

        for i in range(0, r):
            # Propagate forward
            inputs = np.reshape(input_set[i], (1, c))
            hidden_nodes = np.dot(inputs, self.weights1)
            hidden_nodes = np.reshape(hidden_nodes, (1, self.num_hidden))

            # Squash activations
            hidden_nodes = self.sigmoidFunction(hidden_nodes)

            # Add the bias node to hidden layer
            hidden_nodes = np.append(1, hidden_nodes)
            hidden_nodes = np.reshape(hidden_nodes, (1, (self.num_hidden + 1)))
            output = np.dot(hidden_nodes, self.weights2)

            # Squash activations
            output = self.sigmoidFunction(output)

            predicted = np.argmax(output)
            actual = int(output_set[i][0])

            if perform_updates == 1:  # Working on training set, so do back propagation
                # Set up the target vector
                target = np.full((1, 10), .1)
                target[0][int(output_set[i][0])] = .9

                # Calculate error terms
                output_error = output * (1 - output) * (target - output)
                hidden_error = hidden_nodes * (1 - hidden_nodes) * (np.dot(output_error, np.transpose(self.weights2)))

                # Don't need the first element because we dont have weights leading into the bias node in hidden layer
                hidden_error = np.delete(hidden_error, 0)
                hidden_error = np.reshape(hidden_error, (1, self.num_hidden))

                # Update weights
                change_in_w2 = self.changeInWeight(layer=hidden_nodes, delta=output_error, previous=previous_change_w2)
                change_in_w1 = self.changeInWeight(layer=inputs, delta=hidden_error, previous=previous_change_w1)

                self.weights2 += change_in_w2
                self.weights1 += change_in_w1

                # Change previous weights for next iteration
                previous_change_w2 = change_in_w2
                previous_change_w1 = change_in_w1

            else:  # Working on a test set, so save the values to make a confusion matrix
                predicted_list.append(predicted)
                actual_list.append(actual)

            if predicted == actual:
                correct += 1

            total += 1

        accuracy = correct/total
        if perform_updates == 0:
            print(confusion_matrix(actual_list, predicted_list))
        return accuracy

    def storeAccuracy(self, epoch, accuracy, file_name):
        with open(file_name, 'a', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([epoch, accuracy])


training_file = "mnist_train.csv"
testing_file = "mnist_test.csv"
last_accuracy = 0
accuracy_dif = 1
mlp = MLP(training_file=training_file, testing_file=testing_file, learning_rate=.1, num_hidden=100, momentum=.9)
for i in range(0, 51):
    print("Epoch ", i, " confusion matrix")
    testing_acc = mlp.doEpoch(input_set=mlp.testing_input, output_set=mlp.testing_output, perform_updates=0)
    training_acc = mlp.doEpoch(input_set=mlp.training_input, output_set=mlp.training_output, perform_updates=1)
    print("test accuracy = ", testing_acc)
    mlp.storeAccuracy(i, testing_acc, "test_accuracy.csv")
    mlp.storeAccuracy(i, training_acc, "training_accuracy.csv")
    # accuracy_dif = abs(testing_acc - last_accuracy)
    # last_accuracy = testing_acc
    # if accuracy_dif < .001:
    #     break





