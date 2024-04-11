import pennylane as qml
from pennylane import numpy as np
import numpy
from utils.model_functions import performance, cost, accuracy, classifier, prediction
from utils.preprocess import process, get_devices, get_weights, get_layers
import os


class PrunedEnsemble:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_wires = args.num_wires
        self.device_file = args.device_file
        self.devices = get_devices(self.device_file, self.num_wires)
        self.x_train, self.y_train, self.x_valid, self.y_valid = process(self.dataset_name)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_valid.shape[0]
        self.layers = get_layers(self.device_file, self.num_wires)
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.training_epochs=args.training_epochs
        self.features = None
        self.load = args.load
        self.weights = get_weights(self.dataset_name)
        self.num_classes = int(np.unique(self.y_valid).shape[0])

    def run_inference(self):
        var = self.weights
        predictions_valid = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in
                             self.x_valid]
        acc_valid = accuracy(self.y_valid, predictions_valid)

        print("Acc validation: {:0.4f} "
              "".format(acc_valid))
        self.prune()
        var = numpy.load('pruned_weights.npy')
        predictions_valid = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in
                             self.x_valid]
        acc_valid = accuracy(self.y_valid, predictions_valid)

        print("PRUNED Acc validation: {:0.4f} "
              "".format(acc_valid))

    def train(self):
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        if self.load:
            self.prune()
            var = numpy.load('pruned_weights.npy')
            print('Weights Loaded...')
        else:
            var = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        print('Training started...')
        for i in range(self.training_epochs):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, self.num_train, (self.batch_size,))
            valid_batch_index = np.random.randint(0, self.num_test, (200,))
            x_train_batch = self.x_train[batch_index]
            y_train_batch = self.y_train[batch_index]
            var = opt.step(lambda v: cost(v, x_train_batch, y_train_batch), var)
            # Compute predictions on train and validation set
            predictions_valid = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in self.x_valid[valid_batch_index]]

            # Compute accuracy on train and validation set
            acc_valid = accuracy(self.y_valid[valid_batch_index], predictions_valid)

            print("Iter: {:5d} | Cost: {:0.4f} | Acc validation: {:0.4f} "
                  "".format(i + 1, cost(var, self.x_train, self.y_train), acc_valid))

            print("Re pruning testing new accuracy")
            self.prune()
            var = numpy.load('pruned_weights.npy')
            predictions_valid = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in self.x_valid[valid_batch_index]]

            # Compute accuracy on train and validation set
            acc_valid = accuracy(self.y_valid[valid_batch_index], predictions_valid)

            print("Iter: {:5d} | Cost: {:0.4f} | Acc validation: {:0.4f} "
                  "".format(i + 1, cost(var, self.x_train, self.y_train), acc_valid))
            print(var)
            numpy.save('pruned_weights', var)

    def prune(self):
        var = numpy.load('unpruned_weights.npy')
        pruned_val = numpy.argmin(numpy.absolute(var))
        var_flat = numpy.reshape(var, [1, -1])[0]
        var_flat[pruned_val] = 0
        var = numpy.reshape(var, var.shape)
        numpy.save("pruned_weights", var)

