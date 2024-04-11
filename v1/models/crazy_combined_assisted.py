from pennylane import numpy as np
from models.multi_class_ensemble import MultiClassEnsemble
from minimodels.pairs import SmallModel
from minimodels.assistant import AssistedModel
import os
from utils.smallmodel_functions import decision_rule_combo_assist, accuracy_full, accuracy, decision_rule_or, decision_rule_points, decision_rule_combo_points

class CrazyCombinedAssist:
    def __init__(self, args):
        alpha = .875
        alpha_prime = alpha
        self.bitc = MultiClassEnsemble(args)
        self.assis = [AssistedModel(str(i), alpha, is_aux=True) for i in range(8)]
        self.x = np.load(os.path.join('data','splits','full_x.npy'))[1000:1600]
        self.y = np.load(os.path.join('data','splits','full_y.npy'))[1000:1600]
        self.batch_size=30
        self.epochs = 100

    def train(self):
        #self.bitc.train()
        for i in self.assis:
            i.train(self.batch_size, self.epochs)
        print("trained")
        exit()

    def run_inference(self):
        assis = []
        bitc = self.bitc.eval(self.x)
        for i in self.assis:
            assis.append(i.eval(self.x))
        guesses = []

        for b, e0, e1, e2, e3, e4, e5, e6, e7 in zip(bitc, assis[0],assis[1],assis[2],assis[3],assis[4],assis[5], assis[6], assis[7]):
            guesses.append(decision_rule_combo_assist(b,e0, e1, e2, e3, e4, e5, e6, e7,rule=1))
        indecies = []
        guesses_b0 = np.array([i[0] for i in guesses])
        guesses_b1 = np.array([i[1] for i in guesses])
        guesses_b2 = np.array([i[1] for i in guesses])
        for index, i in enumerate(guesses):
            if i[2] ==True:
                indecies.append(index)
        guesses_b0_full_used = guesses_b0[indecies]
        guesses_b1_full_used = guesses_b1[indecies]
        guesses_b2_full_used = guesses_b2[indecies]
        print(len(indecies))
        print(accuracy_full(self.y[indecies], guesses_b0_full_used, guesses_b1_full_used, guesses_b2_full_used))
        print(accuracy_full(self.y, guesses_b0, guesses_b1, guesses_b2))
