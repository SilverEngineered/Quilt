from scipy.stats import mode
import warnings
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
import numpy
import operator
import os

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')


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


def layer(W, num_wires, layer_configuration):
    for j in range(3):
        for i in range(num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 0])



@qml.qnode(qml.device(name='default.qubit', wires=5))
def classifier(weights, features=None, num_wires=5, layer_configuration=1):
    AmplitudeEmbedding(features=features, wires=range(num_wires), normalize=True)
    layer(weights, num_wires, layer_configuration)
    return [qml.expval(qml.PauliZ(0))]

@qml.qnode(qml.device(name='default.qubit', wires=5))
def assisted_classifier(weights, features=None, num_wires=5, layer_configuration=1):
    AmplitudeEmbedding(features=features, wires=range(num_wires), normalize=True)
    for count, W in enumerate(weights):
        layer(W, num_wires,layer_configuration)
    return [qml.expval(qml.PauliZ(0))]

'''
@qml.qnode(qml.device(name='qiskit.ibmq', wires=5, backend='ibmq_manila', ibmqx_token="f75a3efcda7934b18c5ae023d3becc2d00537f13d500bdca5c7844385fa019f9bc84d766098e28d2c836a4c952434b6e2e902f173b5678c1de89bfe776a9ac81"))
def assisted_classifier_real(weights, features=None, num_wires=5, layer_configuration=1):
    AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True)
    for count, W in enumerate(weights):
        layer(W, num_wires, layer_configuration)
    return [qml.expval(qml.PauliZ(0))]
'''

def assisted_classifier_real(weights, features=None, num_wires=5, layer_configuration=1):
    pass


@qml.qnode(qml.device(name='default.qubit', wires=5))
def assisted_classifier_hefty(weights, features=None, num_wires=5, layer_configuration=1):
    AmplitudeEmbedding(features=features, wires=range(num_wires), normalize=True)
    for count, W in enumerate(weights):
        layer(W, num_wires,layer_configuration)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)), qml.expval(qml.PauliZ(4))]
def square_loss(labels, predictions, alpha):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += ((l - p[0]) ** 2)
    loss = loss / len(labels)
    return loss

def square_loss_hefty(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += ((l - p[0]) ** 2)
        loss += ((l - p[1]) ** 2)
        loss += ((l - p[2]) ** 2)
        loss += ((l - p[3]) ** 2)
        loss += ((l - p[4]) ** 2)
    return loss






def square_loss_assisted(labels, predictions, num_qubits=2, alpha=.5):
    return square_loss(labels, predictions, alpha)

def cost(x, features, labels, alpha, layer_configuration=1):
    predictions = [classifier(x, features=f, layer_configuration=layer_configuration) for f in features]
    loss = square_loss(labels, predictions, alpha)
    return loss

def cost_assisted(x, features, labels, alpha, layer_configuration=1):
    predictions = [assisted_classifier(x, features=f, layer_configuration=layer_configuration) for f in features]
    loss = square_loss_assisted(labels, predictions, alpha)
    return loss

def cost_hefty(x, features, labels, alpha, layer_configuration=1):
    predictions = [assisted_classifier_hefty(x, features=f, layer_configuration=layer_configuration) for f in features]
    loss = square_loss_hefty(labels, predictions)
    return loss

def accuracy(labels, predictions):
    acc = 0
    for l, p, in zip(labels, predictions):
        if abs(l - p[0]) < 1e-5 or abs(l - p[1]) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)
    return acc


def accuracy_single(labels, predictions):
    acc = 0
    for l, p, in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)
    return acc

def accuracy_full(labels, predictions_b0, predictions_b1=None, predictions_b2=None):
    acc = 0
    if predictions_b1 is None and predictions_b2 is None:
        for l, b0 in zip(labels, predictions_b0):
            if abs(l - b0) < 1e-5:
                acc = acc + 1
        acc = acc / len(labels)
        return acc
    if predictions_b2 is None:
        for l, b0, b1 in zip(labels, predictions_b0, predictions_b1):
            if abs(l[0] - b0) < 1e-5 and abs(l[1] - b1) < 1e-5:
                acc = acc + 1
        acc = acc / len(labels)
        return acc
    for l, b0, b1, b2 in zip(labels, predictions_b0, predictions_b1, predictions_b2):
        if abs(l[0] - b0) < 1e-5 and abs(l[1] - b1) < 1e-5 and abs(l[2] - b2) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)
    return acc
def prediction(classifier_out):
    sign = np.sign(np.sum([np.sign(i) for i in classifier_out]))
    return sign

def prediction_single(classifier_out):
    return np.sign(classifier_out)

def loader(dataset_name, crazy=False):
    clipped = dataset_name[1:]
    num = dataset_name[0]


    if not crazy:
        if len(dataset_name) > 1:
            x = np.load(os.path.join('data', clipped, 'full_x.npy'))
            y = np.load(os.path.join('data', clipped, num + str('_y.npy')))
            indecies = [c for c,i in enumerate(list(y)) if i == -1]
            indecies_not = [c for c, i in enumerate(list(y)) if i == 1]
            num_cases = len(indecies)
            indecies_not = list(np.array(indecies_not)[np.random.randint(0, len(indecies_not), (num_cases,))])
            full_indecies = np.array(indecies + indecies_not)
            x = x[full_indecies]
            y = y[full_indecies]
        else:
            x = np.load(os.path.join('data', 'splits', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits', dataset_name + str('_y.npy')))
        return x, y
    else:
        if dataset_name == "msb_splits_fashion_4":
            x = np.load(os.path.join('data', 'splits_fashion_4', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion_4', 'full_y.npy'))
            y = [i[0] for i in y]
            return x, y
        if dataset_name == "lsb_splits_fashion_4":
            x = np.load(os.path.join('data', 'splits_fashion_4', 'full_x.npy'))
            y = np.load(os.path.join('data','splits_fashion_4', 'full_y.npy'))
            y = [i[1] for i in y]
            return x, y
        if dataset_name == "msb_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion_8', 'full_y.npy'))
            y = [i[0] for i in y]
            return x, y
        if dataset_name == "mid_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion_8', 'full_y.npy'))
            y = [i[1] for i in y]
            return x, y
        if dataset_name == "lsb_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion_8', 'full_y.npy'))
            y = [i[2] for i in y]
            return x, y

        if dataset_name == "lsb_splits_cifar_4":
            x = np.load(os.path.join('data', 'splits_cifar_4', 'full_x.npy'))
            y = np.load(os.path.join('data','splits_cifar_4', 'full_y.npy'))
            y = [i[1] for i in y]
            return x, y
        if dataset_name == "msb_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar_8', 'full_y.npy'))
            y = [i[0] for i in y]
            return x, y
        if dataset_name == "mid_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar_8', 'full_y.npy'))
            y = [i[1] for i in y]
            return x, y
        if dataset_name == "lsb_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar_8', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar_8', 'full_y.npy'))
            y = [i[2] for i in y]
            return x, y




def weights_save(weights, dataset, epoch, batch_size, accuracy, accs, is_aux):
    if is_aux:
        weights_save_regardless(weights,dataset,epoch,batch_size, accuracy)
    else:
        path_prefix = os.path.join('weights', 'MNIST', 'splits', dataset)
        file_path = os.path.join(path_prefix, 'data.csv')
        weights_path = os.path.join(path_prefix, "weights")
        acc_path = os.path.join(path_prefix, "accs")
        meta_data = "Epoch, Batch_size, Accuracy\n" + str(epoch) + "," + str(batch_size) + "," + str(accuracy)
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
            np.save(weights_path, weights)
            file = open(file_path, "w")
            file.write(meta_data)
            np.save(acc_path, accs)
            file.close()
        file = open(file_path, "r")
        best_acc = float(file.read().split('\n')[1].split(',')[-1])
        file.close()
        if (accuracy > best_acc):
            np.save(weights_path, weights)
            file = open(file_path, "w")
            file.write(meta_data)
            np.save(acc_path, accs)
            file.close()

def weights_save_regardless(weights, dataset, epoch, batch_size, accuracy):
    path_prefix = os.path.join('weights', 'MNIST', 'splits', dataset)
    file_path = os.path.join(path_prefix, 'data.csv')
    weights_path = os.path.join(path_prefix, "weights")
    meta_data = "Epoch, Batch_size, Accuracy\n" + str(epoch) + "," + str(batch_size) + "," + str(accuracy)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    np.save(weights_path, weights)
    file = open(file_path, "w")
    file.write(meta_data)
    file.close()
def flis(num, comp):
    if abs(num - comp) < 1e-5:
        return True
    return False
def flis_r(num1, num2, num3, comp):
    if flis(num1, comp) and flis(num2, comp):
        return True
    if flis(num2, comp) and flis(num3, comp):
        return True
    if flis(num1, comp) and flis(num3, comp):
        return True
    return False
def flis_r_or(num1, num2, num3, comp):
    if flis(num1, comp) or flis(num2, comp) or flis(num3, comp):
        return True
    return False
def decision_rule(m, l, i0, i1, i2, i3, e0, e1, e2, e3):
    if flis(m, -1) and flis(l, -1):
        if flis(i0, -1) and flis(e0, -1):
            return (-1, -1, True)
    if flis(m, -1) and flis(l, 1):
        if flis(i1, -1) and flis(e1, -1):
            return (-1, 1, True)
    if flis(m, 1) and flis(l, -1):
        if flis(i2, -1) and flis(e2, -1):
            return (1, -1, True)
    if flis(m, 1) and flis(l, 1):
        if flis(i3, -1) and flis(e3, -1):
            return (1, 1, True)
    return (m,l, False)

def decision_rule_or(m0, m1, m2, l0, l1, l2, i0, i1, i2, i3, e0, e1, e2, e3):
    if flis_r(m0, m1, m2, -1) and flis_r(l0, l1, l2,-1):
        if flis(i0, -1) or flis(e0, -1):
            return (-1, -1, True)
    if flis_r(m0, m1, m2, -1) and flis_r(l0, l1, l2,1):
        if flis(i1, -1) or flis(e1, -1):
            return (-1, 1, True)
    if flis_r(m0, m1, m2, 1) and flis_r(l0, l1, l2,-1):
        if flis(i2, -1) or flis(e2, -1):
            return (1, -1, True)
    if flis_r(m0, m1, m2, 1) and flis_r(l0, l1, l2,1):
        if flis(i3, -1) or flis(e3, -1):
            return (1, 1, True)

    if flis(i0, -1) and flis(e0, -1) and flis_r_or(m0, m1 ,m2,-1) and flis_r_or(l0, l1 ,l2,-1):
        return (-1, -1, True)
    if flis(i1, -1) and flis(e1, -1) and flis_r_or(m0, m1 ,m2,-1) and flis_r_or(l0, l1 ,l2,1):
        return (-1, 1, True)
    if flis(i1, -1) and flis(e1, -1) and flis_r_or(m0, m1 ,m2,1) and flis_r_or(l0, l1 ,l2,-1):
        return (1, -1, True)
    if flis(i1, -1) and flis(e1, -1) and flis_r_or(m0, m1 ,m2,1) and flis_r_or(l0, l1 ,l2,1):
        return (1, 1, True)

    return (m0,l0, False)
def decision_rule_points(m0, m1, m2, l0, l1, l2, i0, i1, i2, i3, e0, e1, e2, e3):

    if flis_r(m0, m1, m2, -1) and flis_r(l0, l1, l2,-1):
        if flis(i0, -1) and flis(e0, -1):
            return (-1, -1, True)
    if flis_r(m0, m1, m2, -1) and flis_r(l0, l1, l2,1):
        if flis(i1, -1) and flis(e1, -1):
            return (-1, 1, True)
    if flis_r(m0, m1, m2, 1) and flis_r(l0, l1, l2,-1):
        if flis(i2, -1) and flis(e2, -1):
            return (1, -1, True)
    if flis_r(m0, m1, m2, 1) and flis_r(l0, l1, l2,1):
        if flis(i3, -1) and flis(e3, -1):
            return (1, 1, True)



    if flis(e0, -1) and not flis(e1, -1) and not flis(e2, -1) and not flis(e3, -1):
        return (-1, -1, True)
    if not flis(e0, -1) and flis(e1, -1) and not flis(e2, -1) and not flis(e3, -1):
        return (-1, 1, True)
    if not flis(e0, -1) and not flis(e1, -1) and flis(e2, -1) and not flis(e3, -1):
        return (1, -1, True)
    if not flis(e0, -1) and not flis(e1, -1) and not flis(e2, -1) and flis(e3, -1):
        return (1, 1, True)
    points = {'0': 0, '1': 0, '2': 0, '3': 0}
    if flis(m0,1):
        points['2']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['1']+=1
    if flis(m1,1):
        points['2']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['1']+=1
    if flis(m2,1):
        points['2']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['1']+=1
    if flis(l0,1):
        points['1']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['2']+=1
    if flis(l1,1):
        points['1']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['2']+=1
    if flis(l2,1):
        points['1']+=1
        points['3']+=1
    else:
        points['0']+=1
        points['2']+=1
    if flis(i0,-1):
        points['0']+=1
    if flis(i1,-1):
          points['1']+=1
    if flis(i2,-1):
          points['2']+=1
    if flis(i3,-1):
          points['3']+=1
    if flis(e0,-1):
        points['0']+=3
    if flis(e1,-1):
          points['1']+=3
    if flis(e2,-1):
          points['2']+=3
    if flis(e3,-1):
          points['3']+=3
    selection = max(points, key=points.get)
    if selection == "0":
        return (-1, -1, False)
    if selection == "1":
        return (-1, 1, False)
    if selection == "2":
        return (1, -1, False)
    if selection == "3":
        return (1, 1, False)



def decision_rule_combo_assist(b, a0, a1, a2, a3, a4, a5, a6, a7, rule=1):
    if rule ==1:
        if flis(b[0], -1) and flis(b[1], -1) and flis(b[2], -1) and majority(a0):
            return b[0], b[1], b[2], True
        if flis(b[0], -1) and flis(b[1], -1) and flis(b[2], 1) and majority(a1):
            return b[0], b[1], b[2], True
        if flis(b[0], -1) and flis(b[1], 1) and flis(b[2], -1) and majority(a2):
            return b[0], b[1], b[2], True
        if flis(b[0], -1) and flis(b[1], 1) and flis(b[2], 1) and majority(a3):
            return b[0], b[1], b[2], True
        if flis(b[0], 1) and flis(b[1], -1) and flis(b[2], -1) and majority(a4):
            return b[0], b[1], b[2], True
        if flis(b[0], 1) and flis(b[1], -1) and flis(b[2], 1) and majority(a5):
            return b[0], b[1], b[2], True
        if flis(b[0], 1) and flis(b[1], 1) and flis(b[2], -1) and majority(a6):
            return b[0], b[1], b[2], True
        if flis(b[0], 1) and flis(b[1], 1) and flis(b[2], 1) and majority(a7):
            return b[0], b[1], b[2], True
        return decision_rule_combo_points(b, a0, a1, a2, a3, a4, a5, a6, a7)

def decision_rule_combo_points(b, a0, a1, a2, a3, a4, a5, a6, a7):
    cases = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
    b_score = 2
    f_score = 1
    if flis(b[0], -1) and flis(b[1], -1) and flis(b[2], -1):
        cases['0'] +=b_score
    if flis(b[0], -1) and flis(b[1], -1) and flis(b[2], 1):
        cases['1'] += b_score
    if flis(b[0], -1) and flis(b[1], 1) and flis(b[2], -1):
        cases['2'] += b_score
    if flis(b[0], -1) and flis(b[1], 1) and flis(b[2], 1):
        cases['3'] += b_score
    if flis(b[0], 1) and flis(b[1], -1) and flis(b[2], -1):
        cases['4'] += b_score
    if flis(b[0], 1) and flis(b[1], -1) and flis(b[2], 1):
        cases['5'] += b_score
    if flis(b[0], 1) and flis(b[1], 1) and flis(b[2], -1):
        cases['6'] += b_score
    if flis(b[0], 1) and flis(b[1], 1) and flis(b[2], 1):
        cases['7'] += b_score
    for i in a0:
        if np.sign(i) == -1:
            cases['0'] += f_score
    for i in a1:
        if np.sign(i) == -1:
            cases['1'] += f_score
    for i in a2:
        if np.sign(i) == -1:
            cases['2'] += f_score
    for i in a3:
        if np.sign(i) == -1:
            cases['3'] += f_score
    for i in a4:
        if np.sign(i) == -1:
            cases['4'] += f_score
    for i in a5:
        if np.sign(i) == -1:
            cases['5'] += f_score
    for i in a6:
        if np.sign(i) == -1:
            cases['6'] += f_score
    for i in a7:
        if np.sign(i) == -1:
            cases['7'] += f_score
    selection = max(cases, key=cases.get)
    if selection =='0':
        return -1, -1, -1, False
    if selection =='1':
        return -1, -1, 1, False
    if selection =='2':
        return -1, 1, -1, False
    if selection =='3':
        return -1, 1, 1, False
    if selection =='4':
        return 1, -1, -1, False
    if selection =='5':
        return 1, -1, 1, False
    if selection =='6':
        return 1, 1, -1, False
    if selection =='7':
        return 1, 1, 1, False
    else:
        print("ERROR")



def majority(a):
    if np.sum([np.sign(i) for i in a]) <= -1:
        return True
    return False



def decision_rule_combo_assist_2q(b, a0, a1, a2, a3, rule=1):
    if rule ==1:
        if flis(b[0], -1) and flis(b[1], -1) and majority(a0):
            return b[0], b[1], True
        if flis(b[0], -1) and flis(b[1], 1) and majority(a1):
            return b[0], b[1], True
        if flis(b[0], 1) and flis(b[1], -1) and majority(a2):
            return b[0], b[1], True
        if flis(b[0], 1) and flis(b[1], 1) and majority(a3):
            return b[0], b[1], True
        return decision_rule_combo_assist_2q(b, a0, a1, a2, a3, rule=3)
    if rule ==2:
        if flis(b[0], -1) and flis(b[1], -1) and (flis(a0[1], -1) or flis(a0[0], -1)):
            return (b[0], b[1], True)
        if flis(b[0], -1) and flis(b[1], -1) and (flis(a1[1], -1) or flis(a1[0], -1)):
            return (b[0], b[1], True)
        if flis(b[0], -1) and flis(b[1], 1) and (flis(a2[1], -1) or flis(a2[0], -1)):
            return (b[0], b[1], True)
        if flis(b[0], -1) and flis(b[1], 1) and (flis(a3[1], -1) or flis(a3[0], -1)):
            return (b[0], b[1], True)
    if rule ==3:
        cases = {'0': 0, '1': 0, '2': 0, '3': 0}
        b_score = 3
        f_score = 1
        if flis(b[0], -1):
            cases['0'] += b_score
            cases['1'] += b_score
        if flis(b[0], 1):
            cases['2'] += b_score
            cases['3'] += b_score
        if flis(b[1], -1):
            cases['0'] += b_score
            cases['2'] += b_score
        if flis(b[1], 1):
            cases['1'] += b_score
            cases['3'] += b_score
        for i in a0:
            if np.sign(i) == -1:
                cases['0'] += f_score
        for i in a1:
            if np.sign(i) == -1:
                cases['1'] += f_score
        for i in a2:
            if np.sign(i) == -1:
                cases['2'] += f_score
        for i in a3:
            if np.sign(i) == -1:
                cases['3'] += f_score
        selection = max(cases, key=cases.get)
        if selection == '0':
            return -1, -1, False
        if selection == '1':
            return -1, 1, False
        if selection == '2':
            return 1, -1, False
        if selection == '3':
            return 1, 1, False
        else:
            print("ERROR")

def decision_rule_combo_assist_1q(b, a0, a1, rule=1):
    if rule ==1:
        if flis(b, -1) and majority(a0):
            return b,  True
        if flis(b, 1) and majority(a1):
            return b, True
        return decision_rule_combo_assist_1q(b, a0, a1, rule=3)
    if rule ==2:
        if flis(b, -1) and (flis(a0[1], -1) or flis(a0[0], -1)):
            return b, True
        if flis(b, -1) and (flis(a1[1], -1) or flis(a1[0], -1)):
            return b[0], True

    if rule ==3:
        cases = {'0': 0, '1': 0}
        b_score = 2
        f_score = 1
        if flis(b, -1):
            cases['0'] += b_score
        if flis(b, 1):
            cases['1'] += b_score
        for i in a0:
            if np.sign(i) == -1:
                cases['0'] += f_score
        for i in a1:
            if np.sign(i) == -1:
                cases['1'] += f_score
        selection = max(cases, key=cases.get)
        if selection == '0':
            return -1, False
        if selection == '1':
            return 1, False
        else:
            print("ERROR")

def repair(bad_bit, guess, assistants, abs):
    if len(assistants) ==8:
        if bad_bit == 0:
            if guess[1] == -1 and guess[2] == -1:
                if flis(assistants[0], -1):
                    return [-1, guess[1], guess[2]]
                else:
                    return [1, guess[1], guess[2]]
            if guess[1] == -1 and guess[2] == 1:
                if flis(assistants[1], -1):
                    return [-1, guess[1], guess[2]]
                else:
                    return [1, guess[1], guess[2]]
            if guess[1] == 1 and guess[2] == -1:
                if flis(assistants[2], -1):
                    return [-1, guess[1], guess[2]]
                else:
                    return [1, guess[1], guess[2]]
            if guess[1] == 1 and guess[2] == 1:
                if flis(assistants[3], -1):
                    return [-1, guess[1], guess[2]]
                else:
                    return [1, guess[1], guess[2]]
        if bad_bit == 1:
            if guess[0] == -1 and guess[2] == -1:
                if flis(assistants[0], -1):
                    return [guess[0], -1, guess[2]]
                else:
                    return [guess[0], 1, guess[2]]
            if guess[0] == -1 and guess[2] == 1:
                if flis(assistants[1], -1):
                    return [guess[0], -1, guess[2]]
                else:
                    return [guess[0], 1, guess[2]]
            if guess[0] == 1 and guess[2] == -1:
                if flis(assistants[4], -1):
                    return [guess[0], -1, guess[2]]
                else:
                    return [guess[0], 1, guess[2]]
            if guess[0] == 1 and guess[2] == 1:
                if flis(assistants[5], -1):
                    return [guess[0], -1, guess[2]]
                else:
                    return [guess[0], 1, guess[2]]
        if bad_bit == 2:
            if guess[0] == -1 and guess[1] == -1:
                if flis(assistants[0], -1):
                    return [guess[0], guess[1], -1]
                else:
                    return [guess[0], guess[1], 1]
            if guess[0] == -1 and guess[1] == 1:
                if flis(assistants[2], -1):
                    return [guess[0], guess[1], -1]
                else:
                    return [guess[0], guess[1], 1]
            if guess[0] == 1 and guess[1] == -1:
                if flis(assistants[4], -1):
                    return [guess[0], guess[1], -1]
                else:
                    return [guess[0], guess[1], 1]
            if guess[0] == 1 and guess[1] == 1:
                if flis(assistants[6], -1):
                    return [guess[0], guess[1], -1]
                else:
                    return [guess[0], guess[1], 1]



    if len(assistants) ==4:
        if bad_bit == 0:
            if guess[1] == -1:
                if flis(assistants[0], -1):
                    return [-1, guess[1]]
                else:
                    return [1, guess[1]]

            if guess[1] == 1:
                if flis(assistants[1], -1):
                    return [-1, guess[1]]
                else:
                    return [1, guess[1]]
        if bad_bit == 1:
            if guess[0] == -1:
                if flis(assistants[0], -1):
                    return [guess[0], -1]
                else:
                    return [guess[0], 1]
            if guess[0] == 1:
                if flis(assistants[2], -1):
                    return [guess[0], -1]
                elif flis(assistants[3], -1):
                    return [guess[0], 1]
        return guess
        '''
        if bad_bit == 0:
            if guess[1] == -1:
                if np.abs(assistants[0] < np.abs(assistants[2])):
                    return [-1, guess[1]]
                elif np.abs(assistants[0] > np.abs(assistants[2])):
                    return [1, guess[1]]

            if guess[1] == 1:
                if np.abs(assistants[1] < np.abs(assistants[3])):
                    return [-1, guess[1]]
                elif np.abs(assistants[1] > np.abs(assistants[3])):
                    return [1, guess[1]]
        if bad_bit == 1:
            if guess[0] == -1:
                if np.abs(assistants[0] < np.abs(assistants[1])):
                    return [guess[0], -1]
                elif np.abs(assistants[0] > np.abs(assistants[1])):
                    return [guess[0], 1]
            if guess[0] == 1:
                if np.abs(assistants[2] < np.abs(assistants[3])):
                    return [guess[0], -1]
                elif np.abs(assistants[2] > np.abs(assistants[3])):
                    return [guess[0], 1]
        return guess
'''

def consensus_decision(ensemble, assistants, tao):
    all_guesses = []
    bad_bits = 0
    counts = []
    for count, image in enumerate(ensemble):
        assis = [i[count] for i in assistants]
        guess = [np.sign(i) for i in image]
        abs = [numpy.abs(i) for i in image]
        abs_min = min(abs)
        if abs_min < tao:
            bad_bit = numpy.argmin(abs)
            guess = repair(bad_bit, guess, assis, abs)
            bad_bits+=1
            counts.append(count)
        all_guesses.append(guess)
    print("Bad Bits: " + str(bad_bits))
    return all_guesses



