import numpy as np
import os
from matplotlib import pyplot as plt


weights = np.load(os.path.join('..', 'weights', 'splits_cifar_415', 'weights.npy'))


def eval(self, images, real=False, method='base'):
    best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'weights.npy')
    accs = np.load(os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'accs.npy'))
    best_weights = np.load(best_weights_path)
    all_guesses = []
    for image in images:
        guesses = []
        for w in best_weights:
            guesses.append(
                prediction(classifier(w, mixed_layers=self.mixed_layers, features=image, num_qubits=self.num_qubits),
                           8))
        if method == 'partial':
            expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', accs, self.num_qubits)
        if method == 'ensemble':
            expert_guess_weighted = get_expert_guess_method1(guesses, num_qubits=self.num_qubits)
        if method == 'base':
            expert_guess_weighted = guesses[3]
        all_guesses.append(expert_guess_weighted)
    return all_guesses


