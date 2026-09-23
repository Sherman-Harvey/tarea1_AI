"""Entrenar y evaluar el modelo final escogido."""

import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# REEMPLACE estos tres valores por la configuración elegida en la etapa 5.
n = 2
units = 20
learning_rate = 0.01

activation = "relu"
loss = "mse"
batch_size = 64
epochs = 200
patience = 15

dataset = pandas.read_csv("datos_propulsion.csv")

# Nueva división aleatoria. Los hiperparámetros ya fueron escogidos antes.
testset_original = dataset.sample(frac=0.15)
developmentset_original = dataset.drop(testset_original.index)
trainset_original = developmentset_original.sample(frac=0.70 / 0.85)
validationset_original = developmentset_original.drop(trainset_original.index)

trainset_original = trainset_original.reset_index(drop=True)
validationset_original = validationset_original.reset_index(drop=True)
testset_original = testset_original.reset_index(drop=True)

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
testset = testset_original.copy()

for data_normalized, data_original in [
    (trainset, trainset_original),
    (validationset, validationset_original),
    (testset, testset_original)
]:
    data_normalized[features] = (
        data_original[features] - min_features
    ) / difference_features
    data_normalized[targets] = (
        data_original[targets] - min_targets
    ) / difference_targets

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

print("\nCONFIGURACIÓN FINAL")
print("Capas ocultas:", n)
print("Neuronas por capa:", units)
print("Learning rate:", learning_rate)
print("Parámetros entrenables:", network.count_params())

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

train_evaluation = network.evaluate(trainset[features], trainset[targets], verbose=0)
validation_evaluation = network.evaluate(
    validationset[features], validationset[targets], verbose=0
)
test_evaluation = network.evaluate(testset[features], testset[targets], verbose=0)

print("\nRESULTADOS NORMALIZADOS")
print("Epochs realizadas:", len(history.history["loss"]))
print("MSE entrenamiento:", round(train_evaluation[0], 7))
print("MAE entrenamiento:", round(train_evaluation[1], 7))
print("MSE validación:", round(validation_evaluation[0], 7))
print("MAE validación:", round(validation_evaluation[1], 7))
print("MSE prueba:", round(test_evaluation[0], 7))
print("MAE prueba:", round(test_evaluation[1], 7))

predictions_normalized = network.predict(testset[features], verbose=0)
predictions = pandas.DataFrame(predictions_normalized, columns=targets)
predictions[targets] = (
    predictions[targets] * difference_targets
) + min_targets

real_values = testset_original[targets].reset_index(drop=True)
errors = real_values - predictions

metrics_test = pandas.DataFrame({
    "MAE": errors.abs().mean(),
    "RMSE": (errors.pow(2).mean()) ** 0.5,
    "R2": 1 - (
        errors.pow(2).sum()
        / (real_values - real_values.mean()).pow(2).sum()
    )
})

print("\nMÉTRICAS DEL TEST EN UNIDADES ORIGINALES")
print(metrics_test.round(7))

comparison = pandas.DataFrame({
    "kMc real": real_values["kMc"],
    "kMc estimado": predictions["kMc"],
    "kMt real": real_values["kMt"],
    "kMt estimado": predictions["kMt"]
})

print("\nPRIMERAS 20 PREDICCIONES")
print(comparison.head(20).round(7))

history_df = pandas.DataFrame(history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
history_df[["loss", "val_loss"]].plot(ax=axes[0])
axes[0].set_title("Pérdida del modelo final")
axes[0].set_xlabel("Epochs")
axes[0].grid(True, linestyle="--", alpha=0.5)

history_df[["mae", "val_mae"]].plot(ax=axes[1])
axes[1].set_title("MAE del modelo final")
axes[1].set_xlabel("Epochs")
axes[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for position, target in enumerate(targets):
    axes[position].scatter(real_values[target], predictions[target], alpha=0.35)
    minimum = min(real_values[target].min(), predictions[target].min())
    maximum = max(real_values[target].max(), predictions[target].max())
    axes[position].plot(
        [minimum, maximum], [minimum, maximum],
        linestyle="--", color="red"
    )
    axes[position].set_xlabel(target + " real")
    axes[position].set_ylabel(target + " estimado")
    axes[position].set_title("Predicción de " + target)
    axes[position].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

