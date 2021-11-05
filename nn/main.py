import tensorflow as tf
import os
import numpy as np

num_models = 30
mnist_path = os.path.join("..", "data", "MNIST")
x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")
y_train = np.array(np.load(y_train_path))
y_test = np.array(np.load(y_test_path))
x_train = np.reshape(np.load(x_train_path), [-1,28,28])
x_test = np.reshape(np.load(x_test_path), [-1,28,28])



def decision_procedure(all_guesses):
    final_guesses = []
    for i in range(len(all_guesses[0])):
        guesses = [j[i] for j in all_guesses]
        counts = np.bincount(guesses)
        final_guesses.append(np.argmax(counts))
    return final_guesses

def accuracy(guesses, labels):
    total = 0
    correct = 0
    for g,l in zip(guesses,labels):
        total +=1
        if g == l:
            correct +=1
    return correct/total

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


models = [tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(10,activation='relu'),
  tf.keras.layers.Dense(10)
]) for i in range(num_models)]

for model in models:
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
    model.fit(x_train, y_train, epochs=10)

all_guesses = []
accuracies = []
for model in models:
    predictions = model.predict(x_test)
    guesses = [np.argmax(i) for i in predictions]
    all_guesses.append(guesses)
    decision = decision_procedure(all_guesses)
    accuracies.append(accuracy(decision, y_test))
print(accuracies)
