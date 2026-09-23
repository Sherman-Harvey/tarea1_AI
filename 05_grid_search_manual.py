"""Grid Search explicable mediante ciclos anidados."""

import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# Hiperparámetros indicados para el estudio.
n_list = [1, 2, 3]
units_list = [5, 10, 20]
learning_rate_list = [0.01, 0.001]

# Valores que se mantienen fijos para no mezclar demasiadas comparaciones.
activation = "relu"
loss = "mse"
batch_size = 64
epochs = 200
patience = 15

dataset = pandas.read_csv("datos_propulsion.csv")

# Se aparta test, pero no se consulta durante la selección.
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

trainset[features] = (trainset_original[features] - min_features) / difference_features
validationset[features] = (validationset_original[features] - min_features) / difference_features
trainset[targets] = (trainset_original[targets] - min_targets) / difference_targets
validationset[targets] = (validationset_original[targets] - min_targets) / difference_targets


def create_network(n, units, learning_rate):
    """Construye una red con la configuración recibida."""

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

    return network


total_configurations = (
    len(n_list) * len(units_list) * len(learning_rate_list)
)

results = []
idx = 0

print("\nINICIANDO GRID SEARCH")
print("Cantidad de configuraciones:", total_configurations)
print("Todas usan la misma partición en esta ejecución.")

for n in n_list:
    for units in units_list:
        for learning_rate in learning_rate_list:
            idx += 1
            tf.keras.backend.clear_session()

            print(
                "\nConfiguración", idx, "de", total_configurations,
                "- Capas:", n,
                "Neuronas:", units,
                "LR:", learning_rate
            )

            network = create_network(n, units, learning_rate)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            )

            history = network.fit(
                x=trainset[features],
                y=trainset[targets],
                validation_data=(
                    validationset[features],
                    validationset[targets]
                ),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                shuffle=True,
                verbose=0
            )

            validation_evaluation = network.evaluate(
                validationset[features],
                validationset[targets],
                verbose=0
            )

            result = {
                "idx": idx,
                "n": n,
                "units": units,
                "learning_rate": learning_rate,
                "parameters": network.count_params(),
                "epochs_used": len(history.history["loss"]),
                "val_loss": validation_evaluation[0],
                "val_mae": validation_evaluation[1],
                "loss_history": history.history["loss"],
                "val_loss_history": history.history["val_loss"],
                "mae_history": history.history["mae"],
                "val_mae_history": history.history["val_mae"]
            }
            results.append(result)

            print(
                "Val_loss =", round(result["val_loss"], 7),
                "Val_mae =", round(result["val_mae"], 7),
                "Epochs =", result["epochs_used"],
                "Parámetros =", result["parameters"]
            )

results_table = pandas.DataFrame(results)

print_table = results_table[[
    "idx", "n", "units", "learning_rate", "parameters",
    "epochs_used", "val_loss", "val_mae"
]].copy()

print_table = print_table.sort_values(
    by=["val_loss", "parameters"],
    ascending=[True, True]
)

print("\nRESULTADOS ORDENADOS POR MENOR VAL_LOSS")
print(print_table.round(7).to_string(index=False))

best_loss = results_table["val_loss"].min()
limit = best_loss * 1.05

efficient_candidates = results_table[
    results_table["val_loss"] <= limit
][[
    "idx", "n", "units", "learning_rate", "parameters",
    "epochs_used", "val_loss", "val_mae"
]].sort_values(
    by=["parameters", "val_loss", "epochs_used"],
    ascending=[True, True, True]
)

print("\nMODELOS A MENOS DE 5 % DEL MEJOR VAL_LOSS")
print(efficient_candidates.round(7).to_string(index=False))
print(
    "\nLa tabla anterior ayuda a escoger manualmente un modelo que",
    "combine error bajo y pocos parámetros."
)
print("El conjunto de prueba todavía no se evalúa.")

# Primera figura: función de pérdida MSE.
fig, axes = plt.subplots(5, 4, figsize=(16, 18))
axes = axes.flatten()

for position, result in enumerate(results):
    axes[position].plot(result["loss_history"], label="Entrenamiento")
    axes[position].plot(result["val_loss_history"], label="Validación")
    axes[position].set_title(
        "#" + str(result["idx"])
        + " C:" + str(result["n"])
        + " N:" + str(result["units"])
        + " LR:" + str(result["learning_rate"]),
        fontsize=8
    )
    axes[position].set_xlabel("Epochs", fontsize=7)
    axes[position].set_ylabel("MSE", fontsize=7)
    axes[position].grid(True, linestyle="--", alpha=0.5)
    axes[position].legend(fontsize=7)

for position in range(len(results), len(axes)):
    fig.delaxes(axes[position])

plt.tight_layout()
plt.show()

# Segunda figura: error absoluto medio MAE.
fig, axes = plt.subplots(5, 4, figsize=(16, 18))
axes = axes.flatten()

for position, result in enumerate(results):
    axes[position].plot(result["mae_history"], label="Entrenamiento")
    axes[position].plot(result["val_mae_history"], label="Validación")
    axes[position].set_title(
        "#" + str(result["idx"])
        + " C:" + str(result["n"])
        + " N:" + str(result["units"])
        + " LR:" + str(result["learning_rate"]),
        fontsize=8
    )
    axes[position].set_xlabel("Epochs", fontsize=7)
    axes[position].set_ylabel("MAE", fontsize=7)
    axes[position].grid(True, linestyle="--", alpha=0.5)
    axes[position].legend(fontsize=7)

for position in range(len(results), len(axes)):
    fig.delaxes(axes[position])

plt.tight_layout()
plt.show()

