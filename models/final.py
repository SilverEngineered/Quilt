from pennylane import numpy as np
from models.multi_class_ensemble import MultiClassEnsemble
from minimodels.pairs import SmallModel
from minimodels.assistant import AssistedModel
import os
from utils.smallmodel_functions import decision_rule_combo_assist, accuracy_full, decision_rule_combo_assist_2q, decision_rule_combo_assist_1q
from utils.model_functions import get_expert_guess, get_expert_guess_method1
from sklearn import preprocessing
class Full:
    def __init__(self, args):
        self.real = True
        self.eval_real()
        exit()



        #dataset='splits_cifar'
        #dataset = 'splits_cifar_2'
        #dataset = 'splits_cifar_4'


        dataset="8_multiclass"
        #dataset = "splits_2"
        #dataset = "splits_4"


        #dataset="splits_fashion"
        #dataset = "splits_fashion_2"
        #dataset = "splits_fashion_4"

        #dataset = "lsb_splits_8"
        #dataset_shorthand = "splits"
        alpha = .875
        self.crazy = False

        self.args = args
        if "msb" in dataset or "lsb" in dataset or "mid" in dataset:
            self.bitc = MultiClassEnsemble(args, dataset=dataset, crazy=self.crazy, save_name=dataset)
            self.x = np.load(os.path.join('data',dataset_shorthand,'full_x.npy'))[0:1000]
            self.y = np.load(os.path.join('data',dataset_shorthand,'full_y.npy'))[0:1000]
        elif dataset is not "8_multiclass":
            self.bitc = MultiClassEnsemble(args, dataset=dataset)
            num_qubits = self.bitc.num_qubits
            num_classes = int(2 ** num_qubits)
            self.assis = [AssistedModel(str(i) + dataset, alpha, is_aux=True) for i in range(num_classes)]
            self.x = np.load(os.path.join('data',dataset,'full_x.npy'))[0:1000]
            self.y = np.load(os.path.join('data',dataset,'full_y.npy'))[0:1000]
        else:
            self.bitc = MultiClassEnsemble(args)
            num_qubits = self.bitc.num_qubits
            num_classes = int(2 ** num_qubits)
            self.assis = [AssistedModel(str(i), alpha, is_aux=True) for i in range(num_classes)]
            self.x = np.load(os.path.join('data','splits','full_x.npy'))[100:500]
            self.y = np.load(os.path.join('data','splits','full_y.npy'))[100:500]
        self.batch_size=30
        self.epochs = 100
        self.dataset = dataset
    def train(self):
        self.bitc.train()
        for i in self.assis:
            i.train(self.batch_size, self.epochs)
        print("trained")
        exit()

    def run_inference(self):
        if self.crazy:
            self.run_crazy_inference()
            exit()
        self.dataset = "splits_cifar_415"
        if self.dataset == "splits_cifar_415":
            self.x = np.load(os.path.join('data', 'splits_cifar_4', 'full_x.npy'))[0:1000]
            self.y = np.load(os.path.join('data', 'splits_cifar_4', 'full_y.npy'))[0:1000]
            for i in ([1, 3, 10, 15]):
                bitc = self.bitc.eval_15(self.x, i)
                guesses_b0 = np.array([i[0] for i in bitc])
                guesses_b1 = np.array([i[0] for i in bitc])
                ensemble_acc = accuracy_full(self.y, guesses_b0, guesses_b1)
                print("Acc: " + str(i) + " = " + str(ensemble_acc))
        if self.real == True:
            #self.x = np.load(os.path.join('data',self.dataset,'full_x.npy'))[100:150]
            self.x = np.load(os.path.join('data', 'splits', 'full_x.npy'))[100:150]
            self.y = np.load(os.path.join('data', 'splits', 'full_y.npy'))[100:150]
            #self.y = np.load(os.path.join('data',self.dataset,'full_y.npy'))[100:150]
            num_qubits = self.bitc.num_qubits
            # Base
            bitc = self.bitc.eval(self.x, real=True, method='ensemble')
            guesses_b0 = np.array([i[0] for i in bitc])
            if num_qubits == 1:
                ensemble_acc = accuracy_full(self.y, guesses_b0)
                print(ensemble_acc)
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
            counts = [i for i in guesses if i[2] == True]
            indecies = [i for i,g in enumerate(guesses) if g[2] == True]
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
        dataset = "splits_4_last_resort"

        if dataset == "splits_4":
            accs = [32, 24, 34, 18, 36]
            accs = [30, 34, 28, 34, 28]
            self.y = np.load(os.path.join('data', dataset, 'full_y.npy'))[100:150]
            x = []
            for i in range(5):
                x.append(np.sign(np.load(os.path.join('real_results_4', 'splits_4' + str(i) + '_results.npy'))))
            x = np.swapaxes(np.array(x), 0, 1)
            expert_guesses = []
            for guesses in x:
                expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=2))
            guesses0 = [(i[0]) for i in expert_guesses]
            guesses1 = [(i[1]) for i in expert_guesses]
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
            #guesses0 = [i[0] for i in x]
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
                #expert_guesses.append(get_expert_guess_method1(guesses, num_qubits=3))
                expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=3))

            guesses0 = [i[0] for i in expert_guesses]
            guesses1 = [i[1] for i in expert_guesses]
            guesses2 = [i[2] for i in expert_guesses]
            #guesses0 = [(i[0]) for i in x[4]]
            #guesses1 = [i[1] for i in x[4]]
            #guesses2 = [i[2] for i in x[4]]
            full_acc0 = accuracy_full([i[0] for i in self.y], guesses0)
            print("  Acc:" + str(full_acc0))
            full_acc1 = accuracy_full([i[1] for i in self.y], guesses1)
            print("  Acc:" + str(full_acc1))
            full_acc2 = accuracy_full([i[2] for i in self.y], guesses2)
            print("  Acc:" + str(full_acc2))

        if dataset == "splits_4_last_resort":
            accs = [32, 24, 34, 18, 36]
            #accs = [30, 34, 28, 34, 28]
            self.y = np.load(os.path.join('data', 'splits_4', 'full_y.npy'))[100:150]
            x = []
            for i in range(5):
                data = np.load(os.path.join('real_results_4', 'splits_4' + str(i) + '_results.npy'))
                #data = preprocessing.normalize(data, norm='l1')
                min_max_scaler = preprocessing.MinMaxScaler([-1,1])
                data = min_max_scaler.fit_transform(data)
                #data = preprocessing.normalize(data, norm='l1')
                x.append(data)
            x = np.swapaxes(np.array(x), 0, 1)
            expert_guesses = []
            for guesses in x:
                #expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=2))

                expert_guesses.append(get_expert_guess(guesses, 'weighted-partial', accs, num_qubits=2, full=True))
            guesses0 = [np.sign((i[0])) for i in expert_guesses]
            guesses1 = [np.sign((i[1])) for i in expert_guesses]
            full_acc = accuracy_full(self.y, guesses0, guesses1)
            print("  Acc:" + str(full_acc))