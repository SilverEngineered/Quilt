import pennylane as qml
import numpy as np
from tqdm import tqdm
import os

from utils.smallmodel_functions import accuracy, prediction_single, loader, cost, classifier, weights_save, accuracy_single
class SmallModel:
    def __init__(self, dataset, alpha, save_name=None, is_aux=False):
        self.x_train, self.y_train = loader(dataset, crazy=True)
        self.dataset=dataset
        if save_name:
            self.dataset = save_name
        self.opt = qml.NesterovMomentumOptimizer(.5)
        self.weights = 0.01 * np.random.randn(5,5)
        self.alpha = alpha
        self.is_aux = is_aux
    def train(self, batch_size, epochs):
        accs = []
        for i in range(epochs):
            batch_index = np.random.randint(0, len(self.x_train), (batch_size,))
            x_train_batch = self.x_train[batch_index]
            y_train_batch = np.array(self.y_train)[batch_index]
            self.weights = self.opt.step(lambda v: cost(v, x_train_batch, y_train_batch, self.alpha), self.weights)
            accs.append(accuracy_single(y_train_batch, [prediction_single(classifier(self.weights,features=x)) for x in x_train_batch]))
            weights_save(self.weights, self.dataset, i, batch_size, accs[-1], accs[-1], False)
            if i % 50 ==0:
                print("Model: " + self.dataset + " Epoch " + str(i) + " Acc: " + str(np.average(accs)))
                accs = []
    def get_acc(self, batch_size):
        #guesses = [prediction(classifier(self.weights, features=x), num_classes=self.num_classes) for x in self.eval_batch]
        batch_index  = np.random.randint(0, len(self.x_train), (batch_size,))
        x_train_batch = self.x_train[batch_index]
        y_train_batch = self.y_train[batch_index]
        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset, 'weights.npy')
        best_weights = np.load(best_weights_path)
        acc = accuracy(y_train_batch, [prediction(classifier(best_weights,features=x)) for x in x_train_batch])
        return acc
    def eval(self, images):
        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset, 'weights.npy')
        best_weights = np.load(best_weights_path)
        guesses = [prediction_single(classifier(best_weights,features=x)) for x in images]
        return guesses
