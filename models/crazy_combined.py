from pennylane import numpy as np
from models.multi_class_ensemble import MultiClassEnsemble
from minimodels.pairs import SmallModel
import os
from utils.smallmodel_functions import decision_rule_combo_points, accuracy_full, accuracy, decision_rule_or, decision_rule_points

class CrazyCombined:
    def __init__(self, args):
        alpha = .875
        alpha_prime = alpha
        self.bitc = MultiClassEnsemble(args)


        dataset = "fashion_mnist"


        self.x = np.load(os.path.join('data','splits', dataset, 'full_x.npy'))[1000:1900]
        self.y = np.load(os.path.join('data','splits',dataset, 'full_y.npy'))[1000:1900]
        self.batch_size=150
        self.epochs = 150

    def train(self):
        for i in self.bits:
            i.train(self.batch_size, self.epochs)
        for i in self.incs:
            i.train(self.batch_size, self.epochs)
        for j in self.excs:
            j.train(self.batch_size, self.epochs)
        print("trained")
        exit()

    def run_inference(self):
        bitse = []
        incse = []
        excse = []
        for i in self.bits:
            bitse.append(i.eval(self.x))
        for i in self.incs:
            incse.append(i.eval(self.x))
        for j in self.excs:
            excse.append(j.eval(self.x))
        guesses = []
        #for b0, b1, b2, i0, i1, i2, i3, i4, i5, i6, i7, e0, e1, e2, e3, e4, e5, e6, e7 in zip(bitse[0], bitse[1], bitse[2], incse[0], incse[1],incse[2],incse[3],incse[4],incse[5],incse[6],incse[7],excse[0], excse[1], excse[2], excse[3], excse[4], excse[5], excse[6], excse[7]):
        #    guesses.append(decision_rule_combo(b0, b1, b2, i0, i1, i2, i3, i4, i5, i6, i7, e0, e1, e2, e3, e4, e5, e6, e7))
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
