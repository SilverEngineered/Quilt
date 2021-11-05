import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from pennylane.optimize import AdamOptimizer
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
# Read in the dataset
X_train = np.load('../data/MNIST/pca_data/dims_32/digits=[0, 8, 7, 1]/x_train.npy')
X_valid = np.load('../data/MNIST/pca_data/dims_32/digits=[0, 8, 7, 1]/x_test.npy')
Y_train = np.load('../data/MNIST/pca_data/dims_32/digits=[0, 8, 7, 1]/y_train.npy')
Y_valid = np.load('../data/MNIST/pca_data/dims_32/digits=[0, 8, 7, 1]/y_test.npy')
num_train = len(Y_train)
num_wires = 5
num_layers = 5
batch_size = 200
dev = qml.device(name='default.qubit', wires=num_wires)
def layer(W):
    for i in range(num_wires):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 0])
@qml.qnode(dev)
def classifier(weights, features=None):
    AmplitudeEmbedding(features=features, wires=range(num_wires), normalize=True)
    for W in weights:
        layer(W)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + ((l[0]-p[0])**2 + (l[1]-p[1])**2)/2
    loss = loss / len(labels)
    return loss
def cost(x, features, labels):
    predictions = [classifier(x, features=f) for f in features]
    loss = square_loss(labels, predictions)
    return loss
def accuracy(labels, predictions):
    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l[0] - p[0]) < 1e-5 and abs(l[1] - p[1]) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)
    return acc
opt = AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.99, eps=1e-08)
# train the variational classifier
var = 0.01 * np.random.randn(num_layers, num_wires, 3)
for i in range(1000):
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    var = opt.step(lambda v: cost(v, X_train_batch, Y_train_batch), var)
    predictions = [np.sign(classifier(var, features=x)) for x in X_train_batch]
    accur = accuracy(Y_train_batch, predictions)
    print("Iter: {:5d} | Cost: {:0.7f} | Acc: {:0.7f}".format(i + 1, cost(var, X_train_batch, Y_train_batch), accur))
    print(var)
predictions = [np.sign(classifier(var, features=x)) for x in X_train]
accuracy = accuracy(Y_train, predictions)
print(accuracy)
predictions = [np.sign(classifier(var, features=x)) for x in X_valid]
accuracy = accuracy(Y_valid, predictions)
print(accuracy)