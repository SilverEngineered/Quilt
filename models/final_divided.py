from pennylane import numpy as np
import numpy
from models.multi_class_ensemble import MultiClassEnsemble
from minimodels.pairs import SmallModel
from minimodels.assistant import AssistedModel
import os
from utils.smallmodel_functions import decision_rule_combo_assist, accuracy_full, decision_rule_combo_assist_2q, decision_rule_combo_assist_1q, consensus_decision
from utils.model_functions import get_expert_guess, get_expert_guess_method1
from matplotlib import pyplot as plt
class Full_D:
    def __init__(self, args):
        alpha = .5
        self.dataset = args.dataset_name or "8_multiclass"
        self.bitc = MultiClassEnsemble(args)
        self.assis = [AssistedModel(str(i), alpha, is_aux=True) for i in range(4)]
        self.batch_size=40
        self.epochs = args.training_epochs or 200
        self.real = False
    def train(self):
        self.bitc.train()
        for i in self.assis:
            i.train(self.batch_size, self.epochs)
        exit()

    def run_inference(self):
        if self.real:
            self.eval_real_8()
            exit()

        #Last 20% reserved for testing
        self.x = np.load(os.path.join('data', self.dataset, 'full_x.npy'))[-200:]
        self.y = np.load(os.path.join('data', self.dataset, 'full_y.npy'))[-200:]


        assis = [i.eval(self.x, real=False) for i in self.assis]
        for i in (range(5,16)):
            bitc = self.bitc.eval_noisy(self.x, i)
            old_bitc=bitc
            bitc = consensus_decision(bitc, assis, .0005)
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            #guesses_b2 = np.array([i[2] for i in bitc])
            #ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print("Acc: " + str(i) + " = " + str(ensemble_acc))
            bitc = old_bitc
            bitc = consensus_decision(bitc, assis, .0004)
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            #guesses_b2 = np.array([i[2] for i in bitc])
            #ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print("Acc: " + str(i) + " = " + str(ensemble_acc))
            bitc = old_bitc
            bitc = consensus_decision(bitc, assis, .0008)
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            #guesses_b2 = np.array([i[2] for i in bitc])
            #ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print("Acc: " + str(i) + " = " + str(ensemble_acc))
            bitc = old_bitc
            bitc = consensus_decision(bitc, assis, .00095)
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            #guesses_b2 = np.array([i[2] for i in bitc])
            #ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print("Acc: " + str(i) + " = " + str(ensemble_acc))
            bitc = old_bitc
            bitc = consensus_decision(bitc, assis, 0)

            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            #guesses_b2 = np.array([i[2] for i in bitc])
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print("Acc: " + str(i) + " = " + str(ensemble_acc))
            exit()
        num_qubits = self.bitc.num_qubits
        # Base
        bitc = self.bitc.eval(self.x, real=self.real, method='base')
        guesses_b0 = np.array([i[0] for i in bitc])
        if num_qubits ==1:
            base_acc = accuracy_full(self.y, guesses_b0)
        if num_qubits == 2:
            guesses_b1 = np.array([i[1] for i in bitc])
            base_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
        if num_qubits ==3:
            guesses_b1 = np.array([i[1] for i in bitc])
            guesses_b2 = np.array([i[2] for i in bitc])
            base_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
        print(self.dataset + "  Base Acc:" + str(base_acc))

        # Ensemble
        bitc = self.bitc.eval(self.x, real=False, method='ensemble')

        if num_qubits ==1:
            guesses_b0 = np.array([i for i in bitc])
            ensemble_acc = accuracy_full(self.y, guesses_b0)
        if num_qubits == 2:
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
        if num_qubits ==3:
            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            guesses_b2 = np.array([i[2] for i in bitc])
            ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
        print(self.dataset + "  Ensemble Acc:" + str(ensemble_acc))
        # Full
        assis = []
        bitc = self.bitc.eval(self.x, method='partial', real=False)
        for i in self.assis:
            assis.append(i.eval(self.x))
        guesses = []


        guesses_b1 = None
        guesses_b2 = None
        if num_qubits == 1:
            for b, e0, e1 in zip(bitc, assis[0],assis[1]):
                guesses.append(decision_rule_combo_assist_1q(b,e0, e1, rule=1))
            guesses_b0 = np.array([i[0] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0)
            print(self.dataset + "  Full Acc Partial Weighted:" + str(full_acc))
        if num_qubits == 2:
            for b, e0, e1, e2, e3, in zip(bitc, assis[0],assis[1],assis[2],assis[3]):
                guesses.append(decision_rule_combo_assist_2q(b,e0, e1, e2, e3, rule=1))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            counts = [i for i in guesses if i[2]]
            indecies = [i for i,g in enumerate(guesses) if g[2]]
            full_acc = accuracy_full(self.y[indecies], guesses_b0[indecies], guesses_b1[indecies])
            print(self.dataset + "  Full Acc Used:" + str(full_acc))

            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print(self.dataset + "  Full Acc:" + str(full_acc))

            guesses_b0 = np.array([i[0] for i in bitc])
            guesses_b1 = np.array([i[1] for i in bitc])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print(self.dataset + "  Full Acc Partial Weighted:" + str(full_acc))
            exit()
        if num_qubits ==3:
            for b, e0, e1, e2, e3, e4, e5, e6, e7 in zip(bitc, assis[0],assis[1],assis[2],assis[3],assis[4],assis[5], assis[6], assis[7]):
                guesses.append(decision_rule_combo_assist(b,e0, e1, e2, e3, e4, e5, e6, e7,rule=1))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            guesses_b2 = np.array([i[2] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            print(self.dataset + "  Full Acc:" + str(full_acc))

    def run_crazy_inference(self):
        method = 'base'
        self.dataset = 'splits_fashion'
        self.x = np.load(os.path.join('data', self.dataset, 'full_x.npy'))[0:1000]
        self.y = np.load(os.path.join('data', self.dataset, 'full_y.npy'))[0:1000]
        guesses = []
        if self.dataset == "splits_cifar":
            lsb = MultiClassEnsemble(self.args, dataset="lsb_splits_cifar_8", crazy=self.crazy, save_name="lsb_splits_cifar_8").eval(self.x,method=method)
            mid = MultiClassEnsemble(self.args, dataset="mid_splits_cifar_8", crazy=self.crazy, save_name="mid_splits_cifar_8").eval(self.x,method=method)
            msb = MultiClassEnsemble(self.args, dataset="msb_splits_cifar_8", crazy=self.crazy, save_name="msb_splits_cifar_8").eval(self.x,method=method)
            for b0, b1, b2, in zip(lsb, mid, msb):
                guesses.append((b0, b1, b2))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            guesses_b2 = np.array([i[2] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            print(self.dataset + "  Acc:" + str(full_acc) + "  Method: " + str(method))
        if self.dataset == "splits_cifar_4":
            lsb = MultiClassEnsemble(self.args, dataset="lsb_splits_cifar_4", crazy=self.crazy, save_name="lsb_splits_cifar_4").eval(self.x,method=method)
            msb = MultiClassEnsemble(self.args, dataset="msb_splits_cifar_4", crazy=self.crazy, save_name="msb_splits_cifar_4").eval(self.x,method=method)
            for b0, b1, in zip(lsb, msb):
                guesses.append((b0, b1))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print(self.dataset + "  Acc:" + str(full_acc) + "  Method: " + str(method))
        if self.dataset == "splits_fashion":
            lsb = MultiClassEnsemble(self.args, dataset="lsb_splits_fashion_8", crazy=self.crazy, save_name="lsb_splits_fashion_8").eval(self.x,method=method)
            mid = MultiClassEnsemble(self.args, dataset="mid_splits_fashion_8", crazy=self.crazy, save_name="mid_splits_fashion_8").eval(self.x,method=method)
            msb = MultiClassEnsemble(self.args, dataset="msb_splits_fashion_8", crazy=self.crazy, save_name="msb_splits_fashion_8").eval(self.x,method=method)
            for b0, b1, b2, in zip(lsb, mid, msb):
                guesses.append((b0, b1, b2))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            guesses_b2 = np.array([i[2] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            print(self.dataset + "  Acc:" + str(full_acc) + "  Method: " + str(method))
        if self.dataset == "splits":
            lsb = MultiClassEnsemble(self.args, dataset="lsb_splits_8", crazy=self.crazy,
                                     save_name="lsb_splits_8").eval(self.x, method=method)
            mid = MultiClassEnsemble(self.args, dataset="mid_splits_8", crazy=self.crazy,
                                     save_name="mid_splits_fashion_8").eval(self.x, method=method)
            msb = MultiClassEnsemble(self.args, dataset="msb_splits_8", crazy=self.crazy,
                                     save_name="msb_splits_8").eval(self.x, method=method)
            for b0, b1, b2, in zip(msb, mid, lsb):
                guesses.append((b0, b1, b2))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            guesses_b2 = np.array([i[2] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2)
            print(self.dataset + "  Acc:" + str(full_acc) + "  Method: " + str(method))
        if self.dataset == "splits_fashion_4":
            self.x = np.load(os.path.join('data',self.dataset,'full_x.npy'))[0:1000]
            self.y = np.load(os.path.join('data',self.dataset,'full_y.npy'))[0:1000]
            lsb = MultiClassEnsemble(self.args, dataset="lsb_splits_fashion_4", crazy=self.crazy, save_name="lsb_splits_fashion_4").eval(self.x,method=method)
            msb = MultiClassEnsemble(self.args, dataset="msb_splits_fashion_4", crazy=self.crazy, save_name="msb_splits_fashion_4").eval(self.x,method=method)
            for b0, b1, in zip(msb, lsb):
                guesses.append((b0, b1))
            guesses_b0 = np.array([i[0] for i in guesses])
            guesses_b1 = np.array([i[1] for i in guesses])
            full_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
            print(self.dataset + "  Acc:" + str(full_acc) + "  Method: " + str(method))
    def eval_real(self):
        dataset = "splits_4"

        if dataset == "splits_4":
            accs = [32, 24, 34, 18, 36]
            accs = [30, 34, 28, 34, 28]
            self.y = np.load(os.path.join('data', dataset, 'full_y.npy'))[100:150]
            x = []
            for i in range(5):
                x.append(np.sign(np.load(os.path.join('real_results_4', 'splits_4' + str(i) + '_lima__results.npy'))))
            x = np.swapaxes(np.array(x), 0, 1)
            expert_guesses = []
            for guesses in x:
                expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=2))
            guesses0 = [(i[0]) for i in expert_guesses]
            guesses1 = [i[1] for i in expert_guesses]
            full_acc = accuracy_full(self.y,guesses0, guesses1)
            print("  Acc:" + str(full_acc))
        if dataset == "splits_2":
            accs = [1, 1, 1, 1, 1]
            self.y = np.load(os.path.join('data', 'splits_2', 'full_y.npy'))[100:150]
            x = []
            for i in range(4):
                x.append(np.sign(np.load(os.path.join('real_results', 'splits_2' + str(i) + '_lima__results.npy'))))
            x = np.swapaxes(np.array(x), 0, 1)
            expert_guesses = []
            for guesses in x:
                expert_guesses.append(get_expert_guess_method1(guesses, num_qubits=1))
            guesses0 = [i for i in expert_guesses]
            full_acc = accuracy_full(self.y,guesses0)
            print("  Acc:" + str(full_acc))
        if dataset == "splits":
            accs = [18, 14, 22, 18, 14]
            self.y = np.load(os.path.join('data', dataset, 'full_y.npy'))[100:150]
            x = []
            for i in range(5):
                x.append(np.sign(np.load(os.path.join('real_results_8', '8_multiclass' + str(i) + '_results.npy'))))
            x = np.swapaxes(np.array(x), 0, 1)
            expert_guesses = []
            for guesses in x:
                expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=3))
            full_acc = accuracy_full(self.y, expert_guesses)
            print("  Acc:" + str(full_acc))

    def eval_real_8(self):
        self.y = np.load(os.path.join('data', 'splits', 'full_y.npy'))[-200:]
        self.x = np.load(os.path.join('data', 'splits', 'full_x.npy'))[-200:]
        assis = [numpy.load(os.path.join('real_results_assist', str(i) + '_outputs.npy'), allow_pickle=True) for i in range(8)]
        ensemble = [np.load(os.path.join('real_results_8', self.dataset + str(i) + '_lima__results.npy')) for i in range(5)]

        ensemble = numpy.sign(numpy.reshape(ensemble, [50,5,3]))
        expert_guess_weighted = [get_expert_guess(np.sign(i), 'weighted-partial', accuracies=[1,1,1,1,1], num_qubits=3) for i in ensemble]
        accs = [1, 1, 1, 1, 1]
        x = []
        for i in range(5):
            x.append(np.sign(np.load(os.path.join('real_results_8', '8_multiclass' + str(i) + '_cascade_results.npy'))))
        x = np.swapaxes(np.array(x), 0, 1)
        expert_guesses = []
        for guesses in x:
            expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=3))
        full_acc = accuracy_full([i for i in self.y], guesses)
        print("  Acc:" + str(full_acc))