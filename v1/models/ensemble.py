import pennylane as qml
from pennylane import numpy as np
import numpy
from utils.model_functions import performance, cost, accuracy, classifier, prediction
from utils.preprocess import process, get_devices, get_weights, get_layers
import os


class Ensemble:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_wires = args.num_wires
        self.device_file = args.device_file
        self.devices = get_devices(self.device_file, self.num_wires)
        self.x_train, self.y_train, self.x_valid, self.y_valid = process(self.dataset_name)
        self.num_train = self.x_train.shape[0]
        self.layers = get_layers(self.device_file, self.num_wires)
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.features = None
        self.load = args.load
        self.weights = get_weights(self.dataset_name)
        self.num_classes = int(np.unique(self.y_valid).shape[0])

    def run_inference(self):
        self.learners = [self.weights for i in range(len(self.devices))]
        p_valid = []
        for i in range(len(self.x_valid)):
            print('Num Samples              : {:0.0f}\n'.format(i + 1))
            p = []
            for j in range(len(self.learners)):
                qnode = qml.QNode(classifier, self.devices[j])
                p.append(qnode(self.learners[j], layer=self.layers[j], features=self.x_valid[i], num_wires=self.num_wires))
            p_valid.append(p)
            np.save(os.path.join('data', 'dumps', 'P_valid_3_vigo.npy'), p_valid)
            np.save(os.path.join('data', 'dumps', 'Y_valid_3_vigo.npy'), self.y_valid[:i + 1])
            acc, tpr, tnr, fpr, fnr, ppv, npv = performance(self.y_valid[:i + 1], p_valid)
            print('Accuracy                 : {:0.3f}\n'
                  'TPR (Sensitivity, Recall): {:0.3f}\n'
                  'TNR (Specificity)        : {:0.3f}\n'
                  'FPR                      : {:0.3f}\n'
                  'FNR                      : {:0.3f}\n'
                  'PPV (Precision)          : {:0.3f}\n'
                  'NPV                      : {:0.3f}\n'.format(acc, tpr, tnr, fpr, fnr, ppv, npv))

    def train(self):
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        if self.load:
            var = self.weights
        else:
            var = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        print('Training started...')
        for i in range(100):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, self.num_train, (self.batch_size,))
            x_train_batch = self.x_train[batch_index]
            y_train_batch = self.y_train[batch_index]
            var = opt.step(lambda v: cost(v, x_train_batch, y_train_batch), var)
            # Compute predictions on train and validation set
            predictions_train = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in self.x_train]
            predictions_valid = [prediction(classifier(var, features=x), num_classes=self.num_classes) for x in self.x_valid]

            # Compute accuracy on train and validation set
            acc_train = accuracy(self.y_train, predictions_train)
            acc_valid = accuracy(self.y_valid, predictions_valid)

            print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                  "".format(i + 1, cost(var, self.x_train, self.y_train), acc_train, acc_valid))
            print(var)
            numpy.save('weights', var)



