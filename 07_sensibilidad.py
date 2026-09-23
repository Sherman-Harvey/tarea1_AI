"""Sensibilidad ceteris paribus de las entradas."""

import tensorflow as tf
import pandas
import numpy as np
from keras import layers, models, Input
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# Utilizar aquí la misma configuración escogida para el modelo final.
n = 2
units = 20
learning_rate = 0.01

activation = "relu"
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
    loss="mse",
    metrics=["mae"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=patience,
    restore_best_weights=True
)

network.fit(
    x=trainset[features],
    y=trainset[targets],
    validation_data=(validationset[features], validationset[targets]),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=0
)

# Ceteris paribus: todas las entradas quedan en su mediana excepto una.
reference = trainset_original[features].median(axis=0)
quantiles = np.linspace(0.05, 0.95, 21)

sensitivity_rows = []
sensitivity_curves = {}

for feature in features:
    values = trainset_original[feature].quantile(quantiles).drop_duplicates().tolist()

    if len(values) < 2:
        print("Variable constante, no se analiza:", feature)
        continue

    scenarios = pandas.DataFrame([
        reference.to_dict() for value in values
    ])
    scenarios[feature] = values

    scenarios_normalized = scenarios.copy()
    scenarios_normalized[features] = (
        scenarios[features] - min_features
    ) / difference_features

    predictions_normalized = network.predict(
        scenarios_normalized[features], verbose=0
    )
    predictions = pandas.DataFrame(predictions_normalized, columns=targets)
    predictions[targets] = (
        predictions[targets] * difference_targets
    ) + min_targets

    impact_kMc = (
        predictions["kMc"].max() - predictions["kMc"].min()
    ) / difference_targets["kMc"]
    impact_kMt = (
        predictions["kMt"].max() - predictions["kMt"].min()
    ) / difference_targets["kMt"]

    sensitivity_rows.append({
        "Variable": feature,
        "Impacto_relativo_kMc": impact_kMc,
        "Impacto_relativo_kMt": impact_kMt,
        "Impacto_promedio": (impact_kMc + impact_kMt) / 2
    })

    sensitivity_curves[feature] = {
        "values": values,
        "kMc": predictions["kMc"].tolist(),
        "kMt": predictions["kMt"].tolist()
    }

sensitivity_table = pandas.DataFrame(sensitivity_rows)
sensitivity_table = sensitivity_table.sort_values(
    by="Impacto_promedio", ascending=False
).reset_index(drop=True)

print("\nSENSIBILIDAD RELATIVA DE LAS ENTRADAS")
print(sensitivity_table.round(7).to_string(index=False))

selected_features = {}

for target in targets:
    column = "Impacto_relativo_" + target
    most = sensitivity_table.sort_values(column, ascending=False).iloc[0]["Variable"]
    least = sensitivity_table.sort_values(column, ascending=True).iloc[0]["Variable"]
    selected_features[target] = (most, least)

    print("\nSALIDA", target)
    print("Entrada más significativa:", most)
    print("Entrada menos significativa no constante:", least)


def print_change_ranges(feature, target):
    """Muestra dónde cambia más y menos la predicción."""

    values = np.array(sensitivity_curves[feature]["values"], dtype=float)
    outputs = np.array(sensitivity_curves[feature][target], dtype=float)
    changes = np.abs(np.diff(outputs))

    maximum_position = int(np.argmax(changes))
    minimum_position = int(np.argmin(changes))

    print(
        feature, "-", target,
        "mayor cambio entre", round(values[maximum_position], 7),
        "y", round(values[maximum_position + 1], 7)
    )
    print(
        feature, "-", target,
        "menor cambio entre", round(values[minimum_position], 7),
        "y", round(values[minimum_position + 1], 7)
    )


for target in targets:
    most, least = selected_features[target]
    print("\nRANGOS DE CAMBIO PARA", target)
    print_change_ranges(most, target)
    print_change_ranges(least, target)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row, target in enumerate(targets):
    most, least = selected_features[target]

    for column, feature in enumerate([most, least]):
        axes[row, column].plot(
            sensitivity_curves[feature]["values"],
            sensitivity_curves[feature][target],
            marker="o"
        )
        description = "más significativa" if column == 0 else "menos significativa"
        axes[row, column].set_title(target + ": " + feature + " (" + description + ")")
        axes[row, column].set_xlabel(feature)
        axes[row, column].set_ylabel("Predicción de " + target)
        axes[row, column].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

