import pennylane as qml
import numpy as np
from tqdm import tqdm
import os
from utils.smallmodel_functions import accuracy, prediction, loader, cost_assisted, weights_save, cost, assisted_classifier, accuracy_single, assisted_classifier_real

class AssistedModel:
    def __init__(self, dataset, alpha, save_name=None, is_aux=False):
        self.x_train, self.y_train = loader(dataset)
        size = self.x_train.shape[0]
        self.x_test = self.x_train[int(size*.8):]
        self.y_test = self.y_train[int(size * .8):]
        self.x_train = self.x_train[:int(size*.8)]
        self.y_train = self.x_train[:int(size * .8)]

        self.dataset=dataset
        if save_name:
            self.dataset = save_name
        self.opt = qml.NesterovMomentumOptimizer(.5)
        self.weights = 0.01 * np.random.randn(5, 5, 3)
        self.alpha = alpha
        self.is_aux = is_aux
    def train(self, batch_size, epochs):
        accs = []
        for i in range(epochs):
            batch_index = np.random.randint(0, len(self.x_train), (batch_size,))
            x_train_batch = self.x_train[batch_index]
            y_train_batch = self.y_train[batch_index]
            self.weights = self.opt.step(lambda v: cost_assisted(v, x_train_batch, y_train_batch, self.alpha), self.weights)
            acc = accuracy_single(y_train_batch, [prediction(assisted_classifier(self.weights,features=x)) for x in x_train_batch])
            weights_save(self.weights, self.dataset, i, batch_size, acc, 0, self.is_aux)
            if i % 50 ==0:
                print("Model: " + self.dataset + " Epoch " + str(i) + " Acc: " + str(acc))
                accs = []
    def get_acc(self, batch_size):
        #guesses = [prediction(classifier(self.weights, features=x), num_classes=self.num_classes) for x in self.eval_batch]
        batch_index  = np.random.randint(0, len(self.x_train), (batch_size,))
        x_train_batch = self.x_test[batch_index]
        y_train_batch = self.y_test[batch_index]
        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset, 'weights.npy')
        best_weights = np.load(best_weights_path)
        acc = accuracy(y_train_batch, [prediction(assisted_classifier(best_weights,features=x)) for x in x_train_batch])
        return acc
    def eval(self, images, real=False):

        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset, 'weights.npy')
        best_weights = np.load(best_weights_path)
        if real:
            guesses = [assisted_classifier_real(best_weights,features=x) for x in images]
            save_path = os.path.join('real_results_assist', self.dataset + '_outputs.npy')
            np.save(save_path, guesses)
        else:
            guesses = [assisted_classifier(best_weights, features=x) for x in images]
        return guesses
    def eval_noisy(self, images, real=False):
        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset, 'weights.npy')
        best_weights = np.load(best_weights_path)
        if real:
            guesses = [assisted_classifier_real(best_weights,features=x) for x in images]
            save_path = os.path.join('real_results_assist', self.dataset + '_outputs.npy')
            np.save(save_path, guesses)
        else:
            guesses = [assisted_classifier(best_weights, features=x) for x in images]
        return guesses
