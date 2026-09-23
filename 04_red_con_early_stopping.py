"""La misma idea básica, ahora con Early Stopping."""

import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

n = 1
units = 10
activation = "relu"
learning_rate = 0.001
loss = "mse"
batch_size = 64
epochs = 200
patience = 15

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

network.add(layers.Dense(units=len(targets), activation="linear"))

network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=["mae"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=patience,
    restore_best_weights=True
)

history = network.fit(
    x=trainset[features],
    y=trainset[targets],
    validation_data=(validationset[features], validationset[targets]),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

validation_evaluation = network.evaluate(
    validationset[features], validationset[targets], verbose=0
)

print("\nRESULTADOS CON EARLY STOPPING")
print("Máximo permitido de epochs:", epochs)
print("Epochs realmente realizadas:", len(history.history["loss"]))
print("Mejor val_loss observado:", round(min(history.history["val_loss"]), 7))
print("MSE validación restaurado:", round(validation_evaluation[0], 7))
print("MAE validación restaurado:", round(validation_evaluation[1], 7))
print("El conjunto de prueba todavía no se evalúa.")

history_df = pandas.DataFrame(history.history)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
history_df[["loss", "val_loss"]].plot(ax=axes[0])
axes[0].set_title("MSE con Early Stopping")
axes[0].set_xlabel("Epochs realizadas")
axes[0].grid(True, linestyle="--", alpha=0.5)

history_df[["mae", "val_mae"]].plot(ax=axes[1])
axes[1].set_title("MAE con Early Stopping")
axes[1].set_xlabel("Epochs realizadas")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

