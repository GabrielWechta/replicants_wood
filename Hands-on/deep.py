import os

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# history = model.fit(X_train, y_train, epochs=30,
#                     validation_data=(X_valid, y_valid))

# model.evaluate(X_test, y_test)

# housing = fetch_california_housing()
# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     housing.data, housing.target)
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_train_full, y_train_full)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_valid_scaled = scaler.transform(X_valid)
# X_test_scaled = scaler.transform(X_test)
#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
#     keras.layers.Dense(1)
# ])
# model.compile(loss="mean_squared_error", optimizer="sgd")
# history = model.fit(X_train, y_train, epochs=20,
#                     validation_data=(X_valid, y_valid))
# mse_test = model.evaluate(X_test, y_test)
# X_new = X_test[:3]  # pretend these are new instances
# y_pred = model.predict(X_new)
# print(y_pred)

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_01_16-11_28_43'
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
