from scipy.stats import mode
import warnings
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
import numpy
import qiskit.providers.aer.noise as noise

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')


def get_noise_model():
    # Error probabilities
    prob_1 = 0.01  # 1-qubit gate
    #prob_2 = 0.01  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = error_1.tensor(error_1)
    #error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
    return noise_model
def performance(labels, predictions, definition='majority'):
    acc = 0
    fps = 0
    fng = 0
    tps = 0
    tng = 0
    for l, pred in zip(labels, predictions):
        p = mode(np.sign(pred))[0][0]
        if definition == 'averaged':
            p = np.sign(np.mean(pred))
        print(l, pred, p)
        if l == -1 and p == -1:
            tps += 1
            acc += 1
        elif l == -1 and p == 1:
            fng += 1
        elif l == 1 and p == -1:
            fps += 1
        elif l == 1 and p == 1:
            tng += 1
            acc += 1
    acc /= len(labels)
    tpr = 0 if (tps + fng) == 0 else tps / (tps + fng)
    tnr = 0 if (tng + fps) == 0 else tng / (tng + fps)
    fpr = 1 - tnr
    fnr = 1 - tpr
    ppv = 0 if (tps + fps) == 0 else tps / (tps + fps)
    npv = 0 if (tng + fng) == 0 else tng / (tng + fng)
    return acc, tpr, tnr, fpr, fnr, ppv, npv

def layer_new(W):
    for i in range(5):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 0])

def layer(W, num_wires, count=0, mixed_layers=False):
    num_wires = 5
    for i in range(num_wires):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    if mixed_layers == False:
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])
        qml.CNOT(wires=[4, 0])

    else:
        if count %2 == 0:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 0])
            qml.CNOT(wires=[0, 2])
        elif count %3 == 0:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 0])
            qml.CNOT(wires=[0, 1])
        else:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 0])




@qml.qnode(qml.device(name='default.qubit', wires=5))
def classifier(weights, features=None, num_wires=5, mixed_layers=False, num_qubits=3, count=0):
    AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True)
    for W in weights:
        layer(W, num_wires, count, mixed_layers)
    if num_qubits ==3:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]
    elif num_qubits ==2:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
    else:
        return [qml.expval(qml.PauliZ(0))]

@qml.qnode(qml.device(name='qiskit.aer', wires=5, noise_model=get_noise_model()))
def classifier_noisy(weights, features=None, num_wires=5, mixed_layers=False, num_qubits=3, count=0):
    AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True)
    for W in weights:
        layer(W, num_wires, count, mixed_layers)

    if num_qubits ==3:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]
    elif num_qubits ==2:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
    else:
        return [qml.expval(qml.PauliZ(0))]


@qml.qnode(qml.device(name='qiskit.ibmq', wires=5, backend='ibmq_manila', ibmqx_token="f75a3efcda7934b18c5ae023d3becc2d00537f13d500bdca5c7844385fa019f9bc84d766098e28d2c836a4c952434b6e2e902f173b5678c1de89bfe776a9ac81"))
def classifier_real(weights, features=None, num_wires=5, mixed_layers=False, num_qubits=3):
    AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True)
    for count, W in enumerate(weights):
        layer(W, num_wires, count, mixed_layers)
    if num_qubits == 3:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]
    elif num_qubits ==2:
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
    else:
        return [qml.expval(qml.PauliZ(0))]


def square_loss(labels, predictions, num_qubits=3):
    loss = 0
    if num_qubits == 1:
        for l, p in zip(labels, predictions):
            loss = loss + (l - p[0]) ** 2
        loss = loss / len(labels)
        return loss
    elif num_qubits ==2:
        for l, p in zip(labels, predictions):
            loss = loss + ((l[0]-p[0])**2 + (l[1]-p[1])**2)
        loss = loss / len(labels)
        return loss
    else:
        for l, p in zip(labels, predictions):
            loss = loss + ((l[0]-p[0])**2 + (l[1]-p[1])**2+ (l[2]-p[2])**2)
        loss = loss / len(labels)
        return loss

def cost(x, features, labels, num_qubits, mixed_layers, count):
    predictions = [classifier(x, features=f, mixed_layers=mixed_layers, count=count) for f in features]
    loss = square_loss(labels, predictions, num_qubits)
    return loss


def accuracy(labels, predictions, predictions1=None, predictions2=None):
    acc = 0
    if predictions1 is None:
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                acc = acc + 1
        acc = acc / len(labels)
    elif predictions2 is None:
        for l, p, p1, in zip(labels, predictions, predictions1):
            if abs(l[0] - p) < 1e-5 and abs(l[1] - p1) < 1e-5:
                acc = acc + 1
        acc = acc / len(labels)
    else:
        for l, p, p1, p2 in zip(labels, predictions, predictions1, predictions2):
            if abs(l[0] - p) < 1e-5 and abs(l[1] - p1) < 1e-5 and abs(l[2] - p2) < 1e-5:
                acc = acc + 1
        acc = acc / len(labels)
    return acc

def ensemble_accuracy(labels, predictions_each_ensemble, num_qubits):
    each_judge_acc = []
    acc = 0
    # each judge
    for i in predictions_each_ensemble:
        # each sample
        acc = 0
        for j, l in zip(i, labels):
            if num_qubits == 2:
                if abs(l[0] - j[0]) < 1e-5 and abs(l[1] - j[1]) < 1e-5:
                    acc = acc + 1
            else:
                if abs(float(l) - j) < 1e-5:
                    acc = acc + 1
        acc = acc / len(labels)
        each_judge_acc.append(acc)
    return each_judge_acc

def prediction(classifier_out, num_classes):
    if num_classes == 2:
        return np.sign(classifier_out)
    if num_classes == 3:
        if classifier_out > .33:
            return 1
        elif classifier_out < -.33:
            return -1
        else:
            return 0
    else:
        return np.sign(classifier_out)

def prediction_real(classifier_out, num_classes):
    if num_classes == 2:
        return np.sign(classifier_out)
    if num_classes == 3:
        if classifier_out > .33:
            return 1
        elif classifier_out < -.33:
            return -1
        else:
            return 0
    else:
        return np.sign(classifier_out)
def most_common(arr):
    string_list = [str(i) for i in arr]
    most_common = np.array(eval(max(set(string_list), key=string_list.count).replace('.', ',')))
    return most_common

def get_expert_guess(predictions, method, accuracies=None, num_qubits=3, full=False):
    if method == 'unweighted':
        return get_expert_guess_method1(predictions)
    if method == 'weighted':
        pass
    if method == 'unweighted-partial':
        return get_expert_guess_method_3(predictions)
    if method == 'weighted-partial':
        return get_expert_guess_method_4(predictions, accuracies, num_qubits, full)

def get_expert_guess_method1(predictions,num_qubits):
    # num experts, num labels, label arr
    predictions = np.array(predictions)
    if num_qubits == 1:
        counts = {'0': 0, '1': 0}
        for guess in predictions:
            if guess[0] == -1:
                counts['0'] +=1
            if guess[0] == 1:
                counts['1'] += 1
        selection = max(counts, key=counts.get)
        if selection == '0':
            return -1
        if selection == '1':
            return 1
    if num_qubits == 2:
        counts = {'0': 0, '1': 0, '2': 0, '3': 0}
        for guess in predictions:
            if guess[0] == -1 and guess[1] == -1:
                counts['0'] +=1
            if guess[0] == -1 and guess[1] == 1:
                counts['1'] += 1
            if guess[0] == 1 and guess[1] == -1:
                counts['2'] += 1
            if guess[0] == 1 and guess[1] == 1:
                counts['3'] += 1
        selection = max(counts, key=counts.get)
        if selection == '0':
            return -1, -1
        if selection == '1':
            return -1, 1
        if selection == '2':
            return 1, -1
        if selection == '3':
            return 1, 1
    if num_qubits == 3:
        counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for guess in predictions:
            if guess[0] == -1 and guess[1] == -1 and guess[2] == -1:
                counts['0'] +=1
            if guess[0] == -1 and guess[1] == -1 and guess[2] == 1:
                counts['1'] += 1
            if guess[0] == -1 and guess[1] == 1 and guess[2] == -1:
                counts['2'] += 1
            if guess[0] == -1 and guess[1] == 1 and guess[2] == 1:
                counts['3'] += 1
            if guess[0] == 1 and guess[1] == -1 and guess[2] == -1:
                counts['4'] +=1
            if guess[0] == 1 and guess[1] == -1 and guess[2] == 1:
                counts['5'] += 1
            if guess[0] == 1 and guess[1] == 1 and guess[2] == -1:
                counts['6'] += 1
            if guess[0] == 1 and guess[1] == 1 and guess[2] == 1:
                counts['7'] += 1
        selection = max(counts, key=counts.get)
        if selection == '0':
            return -1, -1, -1
        if selection == '1':
            return -1, -1, 1
        if selection == '2':
            return -1, 1, -1
        if selection == '3':
            return -1, 1, 1
        if selection == '4':
            return 1, -1, -1
        if selection == '5':
            return 1, -1, 1
        if selection == '6':
            return 1, 1, -1
        if selection == '7':
            return 1, 1, 1
def get_expert_guess_method_2_single(predictions, accuracies, full):
    # num experts, num labels, label arr
    predictions = np.array(predictions)
    final = 0
    for guess, acc, in zip(predictions, accuracies):
        if acc > .5:
            final += guess*acc
    if not full:
        return np.sign(final)
    else:
        final = 0
        relative_accs = [i/sum(accuracies) for i in accuracies]
        for guess, acc, in zip(predictions, relative_accs):
            final += guess * acc
        return final

def get_expert_guess_method_3(predictions):
    # num experts, num labels, label arr
    predictions = np.array(predictions)
    predictions = numpy.swapaxes(predictions, 0, 1)
    guesses = []
    for count, data_point in enumerate(predictions):
        x = most_common(data_point)
        guesses.append(most_common(data_point))
    return np.array(guesses)

def get_expert_guess_method_4(predictions, accuracies, num_qubits, full):
    # num experts, num labels, label arr
    # single prediction
    predictions = np.array(predictions)
    first_qubits = []
    second_qubits = []
    third_qubits= []
    for expert in predictions:
        first_qubit = expert[0]
        first_qubits.append(first_qubit)
        first_qubits_guesses = get_expert_guess_method_2_single(first_qubits, accuracies, full)
        if num_qubits >1:
            second_qubit = expert[1]
            second_qubits.append(second_qubit)
            second_qubits_guesses = get_expert_guess_method_2_single(second_qubits, accuracies, full)
        if num_qubits > 2:
            third_qubit = expert[2]
            third_qubits.append(third_qubit)
            third_qubits_guesses = get_expert_guess_method_2_single(third_qubits, accuracies, full)

    if num_qubits ==1:
        return first_qubits_guesses
    if num_qubits ==2:
        return first_qubits_guesses, second_qubits_guesses
    if num_qubits ==3:
        return first_qubits_guesses, second_qubits_guesses, third_qubits_guesses

