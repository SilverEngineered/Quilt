from pennylane import numpy as np
import json
import os
import pennylane as qml


def process(dataset="MNIST8",crazy=False):
    path = os.path.join("data", "splits")
    if not crazy:
        num_qubits = 3
        if dataset != "8_multiclass":
            path = os.path.join("data", dataset)
        if dataset == "splits_cifar_2" or dataset == "splits_2" or dataset == "splits_fashion_2":
            num_qubits = 1
        if dataset == "splits_cifar_4" or dataset == "splits_4" or dataset == "splits_fashion_4":
            num_qubits = 2

    else:
        if dataset == "msb_splits_fashion_4":
            x = np.load(os.path.join('data', 'splits_fashion_4', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion_4', 'full_y.npy'))
            y = [i[0] for i in y]
        if dataset == "lsb_splits_fashion_4":
            x = np.load(os.path.join('data', 'splits_fashion_4', 'full_x.npy'))
            y = np.load(os.path.join('data','splits_fashion_4', 'full_y.npy'))
            y = [i[1] for i in y]
        if dataset == "msb_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion', 'full_y.npy'))
            y = [i[0] for i in y]
        if dataset == "mid_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion', 'full_y.npy'))
            y = [i[1] for i in y]
        if dataset == "lsb_splits_fashion_8":
            x = np.load(os.path.join('data', 'splits_fashion', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_fashion', 'full_y.npy'))
            y = [i[2] for i in y]

        if dataset == "msb_splits_cifar_4":
            x = np.load(os.path.join('data', 'splits_cifar_4', 'full_x.npy'))
            y = np.load(os.path.join('data','splits_cifar_4', 'full_y.npy'))
            y = [i[0] for i in y]
        if dataset == "lsb_splits_cifar_4":
            x = np.load(os.path.join('data', 'splits_cifar_4', 'full_x.npy'))
            y = np.load(os.path.join('data','splits_cifar_4', 'full_y.npy'))
            y = [i[1] for i in y]

        if dataset == "msb_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar', 'full_y.npy'))
            y = [i[0] for i in y]
        if dataset == "mid_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar', 'full_y.npy'))
            y = [i[1] for i in y]
        if dataset == "lsb_splits_cifar_8":
            x = np.load(os.path.join('data', 'splits_cifar', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits_cifar', 'full_y.npy'))
            y = [i[2] for i in y]
        if dataset == "msb_splits_8":
            x = np.load(os.path.join('data', 'splits', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits', 'full_y.npy'))
            y = [i[0] for i in y]
        if dataset == "mid_splits_8":
            x = np.load(os.path.join('data', 'splits', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits', 'full_y.npy'))
            y = [i[1] for i in y]
        if dataset == "lsb_splits_8":
            x = np.load(os.path.join('data', 'splits', 'full_x.npy'))
            y = np.load(os.path.join('data', 'splits', 'full_y.npy'))
            y = [i[2] for i in y]
        y = np.array(y)
        return x, y, x, y, 1
    x_train = np.load(os.path.join(path, 'full_x.npy'))
    y_train = np.load(os.path.join(path, 'full_y.npy'))
    x_valid = np.load(os.path.join(path, 'full_x.npy'))
    y_valid = np.load(os.path.join(path, 'full_y.npy'))
    return x_train, y_train, x_valid, y_valid, num_qubits


def get_devices(device_file, num_wires):
    with open(os.path.join('config', device_file), mode='r') as fp:
        data_dict_list = json.load(fp)['devices']
    return [qml.device(**i, wires=num_wires) for i in data_dict_list]


def get_weights(dataset_name):
    if dataset_name == "mushrooms":
        return np.load(os.path.join('weights', 'mushrooms', 'weights.npy'))
    elif dataset_name == "mnist32":
        return np.load(os.path.join('weights', 'MNIST', 'pca_data', 'dims_32', "digits=[1, 5]", "weights.npy"))
    elif dataset_name == "mnist32_1_5_8":
        return np.load(os.path.join('weights', 'MNIST', 'pca_data', 'dims_32', "digits=[1, 5, 8]", "weights.npy"))
    elif dataset_name == "mnist32_0_8_7_1":
        return np.load(os.path.join('weights', 'MNIST', 'pca_data', 'dims_32', "digits=[0, 8, 7, 1]", "weights.npy"))

def get_layers(device_file, num_wires):
    with open(os.path.join('config', device_file), mode='r') as fp:
        data_dict_list = json.load(fp)['layers']
    return [i['type'] for i in data_dict_list]
