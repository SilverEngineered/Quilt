import pennylane as qml
from tqdm import tqdm
from pennylane import numpy as np
import numpy
from utils.model_functions import performance, cost, accuracy, classifier, prediction, ensemble_accuracy, get_expert_guess, classifier_real,get_expert_guess_method_2_single, get_expert_guess_method1, prediction_real, classifier_noisy
from utils.preprocess import process, get_devices, get_weights, get_layers
import os
from utils.smallmodel_functions import weights_save
from tqdm import tqdm
class MultiClassEnsemble:
    def __init__(self, args, dataset="8_multiclass", crazy=False, save_name=None):
        self.dataset_name=dataset
        self.num_wires = args.num_wires
        self.device_file = args.device_file
        #self.devices = get_devices(self.device_file, self.num_wires)


        self.x_train, self.y_train, self.x_valid, self.y_valid, self.num_qubits = process(dataset, crazy)
        size = self.x_train.shape[0]
        self.x_test = self.x_train[int(size*.8):]
        self.y_test = self.y_train[int(size * .8):]
        self.x_train = self.x_train[:int(size*.8)]
        self.y_train = self.x_train[:int(size * .8)]
        self.num_train = self.x_train.shape[0]
        self.num_valid = self.x_valid.shape[0]
        self.layers = get_layers(self.device_file, self.num_wires)
        #self.batch_size = args.batch_size
        #self.batch_size=100
        self.batch_size = 50
        self.num_layers = args.num_layers
        self.features = None
        self.load = args.load
        self.num_epochs = args.training_epochs
        if crazy:
            self.num_epochs = 100
        self.num_classes = 8
        self.experts = 5
        #self.dataset_name += "15"
        self.debug = True
        self.mixed_layers = False

    def train(self):
        accuracies = []
        opts = []
        weights = []
        for i in range(self.experts):
            opts.append(qml.AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.99, eps=1e-08))

        if self.load:
            for i in range(self.experts):
                weights.append(0.01 * np.random.randn(self.num_layers, self.num_wires, 3))
        else:
            for i in range(self.experts):
                weights.append(0.01 * np.random.randn(self.num_layers, self.num_wires, 3))
        print('Training started...')
        pbar = tqdm(range(self.num_epochs))
        item_string = 'Epoch,Method_Number,Members,Accuracy'
        items = [item_string]
        epoch = 0
        for j in pbar:
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, self.num_train, (self.batch_size,))
            batch_index_valid = np.random.randint(0, self.num_valid, (self.batch_size,))
            x_train_batch = np.array(self.x_train[batch_index], requires_grad=False)
            y_train_batch = self.y_train[batch_index]
            x_valid_batch_acc = self.x_valid[batch_index_valid]
            y_valid_batch_acc = self.y_valid[batch_index_valid]
            acc_valid_each_expert = []
            for i in range(self.experts):
                weights[i] = opts[i].step(lambda v: cost(v, x_train_batch, y_train_batch, num_qubits=self.num_qubits, mixed_layers=self.mixed_layers, count=i), weights[i])
                predictions_acc = [prediction(classifier(weights[i], features=x, mixed_layers=self.mixed_layers, num_qubits=self.num_qubits, count=i), num_classes=self.num_classes) for x in x_valid_batch_acc]
                first_qubit_acc = [i[0] for i in predictions_acc]
                second_qubit_acc = None
                third_qubit_acc = None
                if self.num_qubits > 1:
                    second_qubit_acc = [i[1] for i in predictions_acc]
                if self.num_qubits > 2:
                    third_qubit_acc = [i[2] for i in predictions_acc]
                acc_valid_each_expert.append(accuracy(y_valid_batch_acc, first_qubit_acc, second_qubit_acc, third_qubit_acc))
                ensemble_acc = np.average(acc_valid_each_expert)
            weights_save(weights, self.dataset_name, epoch, self.batch_size, ensemble_acc, acc_valid_each_expert,
                             False)
            pbar.set_postfix({'Accuracy': str(ensemble_acc)})
            epoch += 1

    def eval(self, images, real=False, method='base'):
        best_weights_path = os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'weights.npy')
        accs = np.load(os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'accs.npy'))
        best_weights = np.load(best_weights_path)
        all_guesses = []
        all_8 = True
        if real:
            if all_8:
                best_weights_path = os.path.join('weights', 'MNIST', 'splits', '8_multiclass', 'weights.npy')
                accs = np.load(os.path.join('weights', 'MNIST', 'splits', '8_multiclass', 'accs.npy'))
                best_weights = np.load(best_weights_path)
            guesses = []
            np.save(os.path.join('real_results_8', self.dataset_name + '_images.npy'), np.array(images))
            for count, w in enumerate(best_weights):
                guesses = []
                for image in tqdm(images):
                    guesses.append(classifier_real(w, mixed_layers=self.mixed_layers, features=image,num_qubits=self.num_qubits))
                np.save(os.path.join('real_results_8', self.dataset_name + str(count) + '.npy'),np.array(guesses))

            guesses = np.array(guesses)
            images = numpy.swapaxes(guesses, 0, 1)
            for guesses in images:
                expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=self.num_qubits)
            return expert_guess_weighted
        for image in images:
            guesses = []
            for count, w in enumerate(best_weights):
                guesses.append(prediction(classifier(w, mixed_layers=self.mixed_layers,features=image, num_qubits=self.num_qubits, count=count),8))
            if method == 'partial':
                expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', accs, self.num_qubits)
            if method == 'ensemble':
                expert_guess_weighted = get_expert_guess_method1(guesses, num_qubits=self.num_qubits)
            if method == 'base':
                expert_guess_weighted = guesses[3]
            if method == 'full':
                expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', accs, self.num_qubits, full=True)
            all_guesses.append(expert_guess_weighted)
        return all_guesses
    def eval_15(self, images, num):
        self.mixed_layers = False
        weights = np.load(os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'weights.npy'))
        all_guesses = []
        for image in images:
            guesses = []
            for count, w in enumerate(weights[:num+1]):
                #guesses.append(prediction(classifier(w, mixed_layers=self.mixed_layers,features=image, num_qubits=self.num_qubits, count=count),8))
                guesses.append((classifier(w, mixed_layers=self.mixed_layers,features=image, num_qubits=self.num_qubits, count=count)))
            #expert_guess_weighted = get_expert_guess_method1(guesses, num_qubits=self.num_qubits)
            #expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', self.get_acc(), num_qubits=self.num_qubits)
            expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', self.get_acc(), self.num_qubits, full=True)
            all_guesses.append(expert_guess_weighted)
        return all_guesses
    def eval_noisy(self, images, num):
        self.mixed_layers = False
        weights = np.load(os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'weights.npy'))
        all_guesses = []
        for image in images:
            guesses = []
            for count, w in enumerate(weights[:num+1]):
                #guesses.append(prediction(classifier(w, mixed_layers=self.mixed_layers,features=image, num_qubits=self.num_qubits, count=count),8))
                guesses.append((classifier_noisy(w, mixed_layers=self.mixed_layers,features=image, num_qubits=self.num_qubits, count=count)))
            #expert_guess_weighted = get_expert_guess_method1(guesses, num_qubits=self.num_qubits)
            #expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', self.get_acc(), num_qubits=self.num_qubits)
            expert_guess_weighted = get_expert_guess(guesses, 'weighted-partial', self.get_acc(), self.num_qubits, full=True)
            all_guesses.append(expert_guess_weighted)
        return all_guesses
    def get_acc(self):
        accs = np.load(os.path.join('weights', 'MNIST', 'splits', self.dataset_name, 'accs.npy'))
        return accs


