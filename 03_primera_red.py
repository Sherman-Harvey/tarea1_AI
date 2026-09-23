"""Primera red funcional, todavía sin Early Stopping."""

import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# Primera configuración de prueba. Todavía no se considera la mejor.
n = 1
units = 10
activation = "relu"
learning_rate = 0.001
loss = "mse"
batch_size = 64
epochs = 100

dataset = pandas.read_csv("datos_propulsion.csv")

testset_original = dataset.sample(frac=0.15)
developmentset_original = dataset.drop(testset_original.index)
trainset_original = developmentset_original.sample(frac=0.70 / 0.85)
validationset_original = developmentset_original.drop(trainset_original.index)

trainset_original = trainset_original.reset_index(drop=True)
validationset_original = validationset_original.reset_index(drop=True)

min_features = trainset_original[features].min(axis=0)
difference_features = (
    trainset_original[features].max(axis=0) - min_features
).replace(0, 1)

min_targets = trainset_original[targets].min(axis=0)
difference_targets = (
    trainset_original[targets].max(axis=0) - min_targets
).replace(0, 1)

trainset = trainset_original.copy()
validationset = validationset_original.copy()

trainset[features] = (trainset_original[features] - min_features) / difference_features
validationset[features] = (validationset_original[features] - min_features) / difference_features
trainset[targets] = (trainset_original[targets] - min_targets) / difference_targets
validationset[targets] = (validationset_original[targets] - min_targets) / difference_targets

network = models.Sequential()
network.add(Input(shape=(len(features),)))

for i in range(n):
    network.add(layers.Dense(units=units, activation=activation))

# Es regresión con dos salidas continuas, por eso la activación es lineal.
network.add(layers.Dense(units=len(targets), activation="linear"))

network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=["mae"]
)

print("\nRESUMEN DE LA PRIMERA RED")
network.summary()

history = network.fit(
    x=trainset[features],
    y=trainset[targets],
    validation_data=(validationset[features], validationset[targets]),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1
)

train_evaluation = network.evaluate(trainset[features], trainset[targets], verbose=0)
validation_evaluation = network.evaluate(
    validationset[features], validationset[targets], verbose=0
)

print("\nRESULTADOS DE LA PRIMERA RED")
print("MSE entrenamiento:", round(train_evaluation[0], 7))
print("MAE entrenamiento:", round(train_evaluation[1], 7))
print("MSE validación:", round(validation_evaluation[0], 7))
print("MAE validación:", round(validation_evaluation[1], 7))
print("El conjunto de prueba todavía no se evalúa.")

history_df = pandas.DataFrame(history.history)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
history_df[["loss", "val_loss"]].plot(ax=axes[0])
axes[0].set_title("MSE de la primera red")
axes[0].set_xlabel("Epochs")
axes[0].grid(True, linestyle="--", alpha=0.5)

history_df[["mae", "val_mae"]].plot(ax=axes[1])
axes[1].set_title("MAE de la primera red")
axes[1].set_xlabel("Epochs")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

