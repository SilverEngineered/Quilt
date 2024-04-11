import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing


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


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def perform_pca_new(keep_digits=[0,1,2,3,4,5,6,7], pca_dims=32):
    x_train, y_train = load_mnist(os.path.join('..','data','fashion_mnist'),'train')
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    x_train = x_train[train_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    #-----------------------------------------------------------------------------------------
    y_path = os.path.join("..", "data", "splits_fashion", "bits")
    np.save(y_path, train_amplitudes)
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1 and i[2] ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "1_y")
    np.save(y_path, split_1_y)
    #-----------------------------------------------------------------------------------------
    split_2_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==-1:
            split_2_y.append(-1)
        else:
            split_2_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "2_y")
    np.save(y_path, split_2_y)
    #-----------------------------------------------------------------------------------------
    split_3_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1 and i[2] ==1:
            split_3_y.append(-1)
        else:
            split_3_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "3_y")
    np.save(y_path, split_3_y)
    #-----------------------------------------------------------------------------------------
    split_4_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==-1:
            split_4_y.append(-1)
        else:
            split_4_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "4_y")
    np.save(y_path, split_4_y)
    #-----------------------------------------------------------------------------------------
    split_5_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1 and i[2] ==1:
            split_5_y.append(-1)
        else:
            split_5_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "5_y")
    np.save(y_path, split_5_y)
    #-----------------------------------------------------------------------------------------
    split_6_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==-1:
            split_6_y.append(-1)
        else:
            split_6_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "6_y")
    np.save(y_path, split_6_y)
    #-----------------------------------------------------------------------------------------
    split_7_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1 and i[2] ==1:
            split_7_y.append(-1)
        else:
            split_7_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion", "7_y")
    np.save(y_path, split_7_y)
    #-----------------------------------------------------------------------------------------
    x_path = os.path.join("..", "data", "splits_fashion", "full_x")
    y_path = os.path.join("..", "data", "splits_fashion", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)

def perform_pca_4(keep_digits=[0,1,2,3], pca_dims=32):
    x_train, y_train = load_mnist(os.path.join('..','data','fashion_mnist'),'train')
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    x_train = x_train[train_indecies]
    pca = PCA(pca_dims)
    pca.fit(preprocessing.normalize(x_train))
    x_train_transformed = scale_data(pca.transform(preprocessing.normalize(x_train)))
    #-----------------------------------------------------------------------------------------
    split_0_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==-1:
            split_0_y.append(-1)
        else:
            split_0_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion_4", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==-1 and i[1] ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion_4", "1_y")
    np.save(y_path, split_1_y)
    #-----------------------------------------------------------------------------------------
    split_2_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==-1:
            split_2_y.append(-1)
        else:
            split_2_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion_4", "2_y")
    np.save(y_path, split_2_y)
    #-----------------------------------------------------------------------------------------
    split_3_y = []
    for index, i in enumerate(train_amplitudes):
        if i[0] ==1 and i[1] ==1:
            split_3_y.append(-1)
        else:
            split_3_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion_4", "3_y")
    np.save(y_path, split_3_y)
    x_path = os.path.join("..", "data", "splits_fashion_4", "full_x")
    y_path = os.path.join("..", "data", "splits_fashion_4", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)

def perform_pca_2(keep_digits=[0,1], pca_dims=32):
    x_train, y_train = load_mnist(os.path.join('..','data','fashion_mnist'),'train')
    amplitude_map = get_amplitude_map(keep_digits)
    train_indecies, train_amplitudes = get_indecies_amplitudes(amplitude_map, y_train)
    x_train = x_train[train_indecies]
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
    y_path = os.path.join("..", "data", "splits_fashion_2", "0_y")
    np.save(y_path, split_0_y)
    #-----------------------------------------------------------------------------------------
    split_1_y = []
    for index, i in enumerate(train_amplitudes):
        if i ==1:
            split_1_y.append(-1)
        else:
            split_1_y.append(1)
    y_path = os.path.join("..", "data", "splits_fashion_2", "1_y")
    np.save(y_path, split_1_y)
    x_path = os.path.join("..", "data", "splits_fashion_2", "full_x")
    y_path = os.path.join("..", "data", "splits_fashion_2", "full_y")
    np.save(x_path, x_train_transformed)
    np.save(y_path, train_amplitudes)
if __name__ == '__main__':
    perform_pca_2()
    perform_pca_4()