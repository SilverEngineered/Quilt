import pennylane as qml
from tqdm import tqdm
from pennylane import numpy as np
import numpy
from minimodels.pairs import SmallModel
import os
from utils.smallmodel_functions import decision_rule, accuracy_full, accuracy, decision_rule_or, decision_rule_points

class Crazy:
    def __init__(self, args):
        alpha = .75
        alpha_prime = 1 - alpha
        self.dataset = "splits_fashion_4"
        self.pair0 = SmallModel("msb_" + self.dataset, 1)
        self.pair1 = SmallModel("lsb_" + self.dataset, 1)
        self.pair00 = SmallModel("msb_" + self.dataset, 1, save_name="msb_" + self.dataset + "_1")
        self.pair11 = SmallModel("lsb_" + self.dataset, 1, save_name="lsb" + self.dataset+ "_1")
        self.pair000 = SmallModel("msb_" + self.dataset, 1, save_name="msb_" + self.dataset+ "_2")
        self.pair111 = SmallModel("lsb_" + self.dataset, 1,  save_name="lsb_" + self.dataset+ "_2")



        '''self.inc0  = SmallModel("0", 1, alpha)
        self.inc1  = SmallModel("1", 1, alpha)
        self.inc2  = SmallModel("2", 1, alpha)
        self.inc3  = SmallModel("3", 1, alpha)
        self.exc0  = SmallModel("0", 1, alpha_prime, save_name="0-")
        self.exc1  = SmallModel("1", 1, alpha_prime, save_name="1-")
        self.exc2  = SmallModel("2", 1, alpha_prime, save_name="2-")
        self.exc3  = SmallModel("3", 1, alpha_prime, save_name="3-")'''
        self.x = np.load(os.path.join('data', self.dataset, 'full_x.npy'))[:1600]
        self.y = np.load(os.path.join('data', self.dataset, 'full_y.npy'))[:1600]
        self.batch_size=100
        self.epochs = 200

    def train(self):
        self.pair0.train(self.batch_size, self.epochs)
        self.pair1.train(self.batch_size, self.epochs)
        self.pair00.train(self.batch_size, self.epochs)
        self.pair11.train(self.batch_size, self.epochs)
        self.pair000.train(self.batch_size, self.epochs)
        self.pair111.train(self.batch_size, self.epochs)
        '''self.inc0.train(self.batch_size, self.epochs)
        self.inc1.train(self.batch_size, self.epochs)
        self.inc2.train(self.batch_size, self.epochs)
        self.inc3.train(self.batch_size, self.epochs)
        self.exc0.train(self.batch_size, self.epochs)
        self.exc1.train(self.batch_size, self.epochs)
        self.exc2.train(self.batch_size, self.epochs)
        self.exc3.train(self.batch_size, self.epochs)'''
        print("trained")
        exit()

    def run_inference(self):
        msb_0 = self.pair0.eval(self.x)
        lsb_0 = self.pair1.eval(self.x)
        msb_1 = self.pair00.eval(self.x)
        lsb_1 = self.pair11.eval(self.x)
        msb_2 = self.pair000.eval(self.x)
        lsb_2 = self.pair111.eval(self.x)
        '''inc_0 = self.inc0.eval(self.x)
        inc_1 = self.inc1.eval(self.x)
        inc_2 = self.inc2.eval(self.x)
        inc_3 = self.inc3.eval(self.x)
        exc_0 = self.exc0.eval(self.x)
        exc_1 = self.exc1.eval(self.x)
        exc_2 = self.exc2.eval(self.x)
        exc_3 = self.exc3.eval(self.x)'''
        guesses = []
        for m0, m1, m2, l0, l1, l2, i0, i1, i2, i3, e0, e1, e2, e3 in zip(msb_0, lsb_0, msb_1, lsb_1,msb_2, lsb_2, inc_0, inc_1, inc_2, inc_3, exc_0, exc_1, exc_2, exc_3):
            guesses.append(decision_rule_points(m0, m1, m2, l0, l1, l2, i0, i1, i2, i3, e0, e1, e2, e3))
        indecies = []
        guesses_b0 = np.array([i[0] for i in guesses])
        guesses_b1 = np.array([i[1] for i in guesses])
        for index, i in enumerate(guesses):
            if i[2] ==True:
                indecies.append(index)
        guesses_b0_full_used = guesses_b0[indecies]
        guesses_b1_full_used = guesses_b1[indecies]
        print(len(indecies))
        print(accuracy_full(self.y[indecies], guesses_b0_full_used, guesses_b1_full_used))
        print(accuracy_full(self.y, guesses_b0, guesses_b1))
