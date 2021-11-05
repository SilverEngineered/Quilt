import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt

def scale_data(data, scale=[0, 1], dtype=np.float32):
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)


def get_indecies_amplitudes(amplitude_map, labels):
    indecies = []
    amplitudes = []
    for i in range(labels.shape[0]):
        if labels[i] in amplitude_map:
            indecies.append(i)
            amplitudes.append(amplitude_map[labels[i]])
    return indecies, amplitudes


def get_amplitude_map(keep_digits):
    length = len(keep_digits)
    map = {}
    if length == 2:
        map[keep_digits[0]] = -1
        map[keep_digits[1]] = 1
    if length == 3:
        map[keep_digits[0]] = -1
        map[keep_digits[1]] = 0
        map[keep_digits[2]] = 1
    if length == 4:
        map[keep_digits[0]] = [-1, -1]
        map[keep_digits[1]] = [-1, 1]
        map[keep_digits[2]] = [1, -1]
        map[keep_digits[3]] = [1, 1]
    if length == 8:
        map[keep_digits[0]] = [-1, -1, -1]
        map[keep_digits[1]] = [-1, -1, 1]
        map[keep_digits[2]] = [-1, 1, -1]
        map[keep_digits[3]] = [-1, 1, 1]
        map[keep_digits[4]] = [1, -1, -1]
        map[keep_digits[5]] = [1, -1, 1]
        map[keep_digits[6]] = [1, 1, -1]
        map[keep_digits[7]] = [1, 1, 1]
    return map


def perform_pca(keep_digits=list(range(8)), pca_dims=32):
    mnist_path = os.path.join("..", "data", "MNIST")
    x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
    x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
    y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
    y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
    y_train = np.array(np.load(y_train_path), dtype=float)
    y_test = np.array(np.load(y_test_path), dtype=float)
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    test_indecies, test_amplitudes = get_indecies_amplitudes(amplitude_map, y_test)
    x_train = np.load(x_train_path)[train_indecies]
    x_test = np.load(x_test_path)[test_indecies]
    pcas = []
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    approximation = pca.inverse_transform(pca.transform(x_train))
    plt.imshow(approximation[0].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest', clim=(0, 255))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join('', "reconstruct.pdf"),bbox_inches = 'tight',
    pad_inches = 0)
    plt.close()
    exit()
    x_train_transformed = pca.transform(preprocessing.normalize(x_train))
    x_test_transformed = pca.transform(preprocessing.normalize(x_test))
    x_train_transformed = (x_train_transformed - np.min(x_train_transformed)) / (np.max(x_train_transformed) - np.min(x_train_transformed))
    path = os.path.join("..", "data", "MNIST", "pca_data", "dims_" + str(pca_dims), "digits=" + str(keep_digits))
    if not os.path.exists(path):
        os.makedirs(path)
    x_train_path = os.path.join(path, 'x_train')
    x_test_path = os.path.join(path, 'x_test')
    y_train_path = os.path.join(path, 'y_train')
    y_test_path = os.path.join(path, 'y_test')
    np.save(x_train_path, scale_data(x_train_transformed))
    np.save(x_test_path, scale_data(x_test_transformed))
    np.save(y_train_path, train_amplitudes)
    np.save(y_test_path, test_amplitudes)
    approximation = pca.inverse_transform(pca.transform(x_train))
    plt.imshow(approximation[0].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest', clim=(0, 255))
    plt.title("Reconstruction of MNIST Image with 32 Principal Components")
    plt.savefig(os.path.join('', "reconstruct.pdf"))
    plt.close()

def perform_pca_new(keep_digits=[0,1,2,3,4,5,6,7], pca_dims=32):
    mnist_path = os.path.join("..", "data", "MNIST")
    x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
    x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
    y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
    y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
    y_train = np.array(np.load(y_train_path), dtype=float)
    y_test = np.array(np.load(y_test_path), dtype=float)
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    test_indecies, test_amplitudes = get_indecies_amplitudes(amplitude_map, y_test)
    x_train = np.load(x_train_path)[train_indecies]
    x_test = np.load(x_test_path)[test_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    #-----------------------------------------------------------------------------------------
    b0 = []
    b1 = []
    b2 = []
    for i in train_amplitudes:
        b0.append(i[0])
        b1.append(i[1])
        b2.append(i[2])
    y_path = os.path.join("..", "data", "splits", "b0")
    np.save(y_path, b0)
    y_path = os.path.join("..", "data", "splits", "b1")
    np.save(y_path, b1)
    y_path = os.path.join("..", "data", "splits", "b2")
    np.save(y_path, b2)
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits", "1_y")
    np.save(y_path, split_1_y)
    #-----------------------------------------------------------------------------------------
    split_2_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==-1:
            split_2_y.append(-1)
        else:
            split_2_y.append(1)
    y_path = os.path.join("..", "data", "splits", "2_y")
    np.save(y_path, split_2_y)
    #-----------------------------------------------------------------------------------------
    split_3_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==1:
            split_3_y.append(-1)
        else:
            split_3_y.append(1)
    y_path = os.path.join("..", "data", "splits", "3_y")
    np.save(y_path, split_3_y)
    #-----------------------------------------------------------------------------------------
    split_4_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==-1:
            split_4_y.append(-1)
        else:
            split_4_y.append(1)
    y_path = os.path.join("..", "data", "splits", "4_y")
    np.save(y_path, split_4_y)
    #-----------------------------------------------------------------------------------------
    split_5_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==1:
            split_5_y.append(-1)
        else:
            split_5_y.append(1)
    y_path = os.path.join("..", "data", "splits", "5_y")
    np.save(y_path, split_5_y)
    #-----------------------------------------------------------------------------------------
    split_6_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==-1:
            split_6_y.append(-1)
        else:
            split_6_y.append(1)
    y_path = os.path.join("..", "data", "splits", "6_y")
    np.save(y_path, split_6_y)
    #-----------------------------------------------------------------------------------------
    split_7_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==1:
            split_7_y.append(-1)
        else:
            split_7_y.append(1)
    y_path = os.path.join("..", "data", "splits", "7_y")
    np.save(y_path, split_7_y)
    #-----------------------------------------------------------------------------------------
    x_path = os.path.join("..", "data", "splits", "full_x")
    y_path = os.path.join("..", "data", "splits", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)


def perform_pca_multi(keep_digits=[0,1,2,3,4,5,6,7], pca_dims=32):
    mnist_path = os.path.join("..", "data", "MNIST")
    x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
    x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
    y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
    y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
    y_train = np.array(np.load(y_train_path), dtype=float)
    y_test = np.array(np.load(y_test_path), dtype=float)
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    test_indecies, test_amplitudes = get_indecies_amplitudes(amplitude_map, y_test)
    x_train = np.load(x_train_path)[train_indecies]
    x_test = np.load(x_test_path)[test_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    #-----------------------------------------------------------------------------------------
    b0 = []
    b1 = []
    b2 = []
    for i in train_amplitudes:
        b0.append(i[0])
        b1.append(i[1])
        b2.append(i[2])
    y_path = os.path.join("..", "data", "splits", "b0")
    np.save(y_path, b0)
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits", "1_y")
    np.save(y_path, split_1_y)
    #-----------------------------------------------------------------------------------------
    split_2_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==-1:
            split_2_y.append(-1)
        else:
            split_2_y.append(1)
    y_path = os.path.join("..", "data", "splits", "2_y")
    np.save(y_path, split_2_y)
    #-----------------------------------------------------------------------------------------
    split_3_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==1:
            split_3_y.append(-1)
        else:
            split_3_y.append(1)
    y_path = os.path.join("..", "data", "splits", "3_y")
    np.save(y_path, split_3_y)
    #-----------------------------------------------------------------------------------------
    split_4_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==-1:
            split_4_y.append(-1)
        else:
            split_4_y.append(1)
    y_path = os.path.join("..", "data", "splits", "4_y")
    np.save(y_path, split_4_y)
    #-----------------------------------------------------------------------------------------
    split_5_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==1:
            split_5_y.append(-1)
        else:
            split_5_y.append(1)
    y_path = os.path.join("..", "data", "splits", "5_y")
    np.save(y_path, split_5_y)
    #-----------------------------------------------------------------------------------------
    split_6_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==-1:
            split_6_y.append(-1)
        else:
            split_6_y.append(1)
    y_path = os.path.join("..", "data", "splits", "6_y")
    np.save(y_path, split_6_y)
    #-----------------------------------------------------------------------------------------
    split_7_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==1:
            split_7_y.append(-1)
        else:
            split_7_y.append(1)
    y_path = os.path.join("..", "data", "splits", "7_y")
    np.save(y_path, split_7_y)
    #-----------------------------------------------------------------------------------------
    x_path = os.path.join("..", "data", "splits", "full_x")
    y_path = os.path.join("..", "data", "splits", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)



def perform_pca_4(keep_digits=[0,8, 7, 1], pca_dims=16):
    mnist_path = os.path.join("..", "data", "MNIST")
    x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
    x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
    y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
    y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
    y_train = np.array(np.load(y_train_path), dtype=float)
    y_test = np.array(np.load(y_test_path), dtype=float)
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    test_indecies, test_amplitudes = get_indecies_amplitudes(amplitude_map, y_test)
    x_train = np.load(x_train_path)[train_indecies]
    x_test = np.load(x_test_path)[test_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    '''
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits_4", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits_4", "1_y")
    np.save(y_path, split_1_y)
    #-----------------------------------------------------------------------------------------
    split_2_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1:
            split_2_y.append(-1)
        else:
            split_2_y.append(1)
    y_path = os.path.join("..", "data", "splits_4", "2_y")
    np.save(y_path, split_2_y)
    #-----------------------------------------------------------------------------------------
    split_3_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1:
            split_3_y.append(-1)
        else:
            split_3_y.append(1)
    y_path = os.path.join("..", "data", "splits_4", "3_y")
    np.save(y_path, split_3_y)
    '''
    x_path = os.path.join("..", "data", "splits_4_small", "full_x")
    y_path = os.path.join("..", "data", "splits_4_small", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)


def perform_pca_2(keep_digits=[0,1], pca_dims=32):
    mnist_path = os.path.join("..", "data", "MNIST")
    x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
    x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
    y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
    y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
    y_train = np.array(np.load(y_train_path), dtype=float)
    y_test = np.array(np.load(y_test_path), dtype=float)
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    test_indecies, test_amplitudes = get_indecies_amplitudes(amplitude_map, y_test)
    x_train = np.load(x_train_path)[train_indecies]
    x_test = np.load(x_test_path)[test_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits_2", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits_2", "1_y")
    np.save(y_path, split_1_y)

    x_path = os.path.join("..", "data", "splits_2", "full_x")
    y_path = os.path.join("..", "data", "splits_2", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)

if __name__ == '__main__':
    keep_digits = [0,1, 2, 3, 4, 5, 6, 7]
    pca_dims = 32
    perform_pca(keep_digits, pca_dims)
    #perform_pca_4()
