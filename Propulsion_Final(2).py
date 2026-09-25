"""Programa completo para estimar kMc y kMt mediante una red neuronal.

1. Lectura y análisis descriptivo.
2. División aleatoria en entrenamiento, validación y prueba.
3. Normalización min-max sin fuga de información.
4. Grid Search manual de capas, neuronas y learning rate.
5. Selección de hiperparámetros digitada por el usuario.
6. Entrenamiento y evaluación del modelo final.
7. Análisis local de sensibilidad.

"""

import tensorflow as tf
import pandas
import numpy as np
from keras import layers, models, Input
import matplotlib.pyplot as plt


# Entradas y salidas suministradas en el dataset.
features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# Porcentajes para la división aleatoria.
train_percentage = 0.70
validation_percentage = 0.15
test_percentage = 0.15

# Hiperparámetros que se estudiarán en el Grid Search.
n_list = [1, 2, 3]
units_list = [5, 10, 20]
learning_rate_list = [0.01, 0.005, 0.001]

# Valores que permanecen fijos durante la comparación.
activation = "relu"
loss = "mse"
batch_size = 64
epochs = 200
patience = 15


def create_network(n, units, learning_rate):
    """Construye y compila una red con la configuración indicada."""

    network = models.Sequential()
    network.add(Input(shape=(len(features),)))

    for i in range(n):
        network.add(layers.Dense(units=units, activation=activation))

    # kMc y kMt son valores continuos, por eso se usa salida lineal.
    network.add(layers.Dense(units=len(targets), activation="linear"))

    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["mae"]
    )

    return network


def request_integer(message, allowed_values):
    """Solicita un entero hasta que pertenezca a la lista estudiada."""

    while True:
        try:
            value = int(input(message))
            if value in allowed_values:
                return value
        except ValueError:
            pass

        print("Valor no válido. Opciones permitidas:", allowed_values)


def request_float(message, allowed_values):
    """Solicita un decimal hasta que pertenezca a la lista estudiada."""

    while True:
        try:
            value = float(input(message).replace(",", "."))
            if any(np.isclose(value, option) for option in allowed_values):
                return value
        except ValueError:
            pass

        print("Valor no válido. Opciones permitidas:", allowed_values)


# =====================================================================
# 1: LECTURA Y ANÁLISIS DEL DATASET
# =====================================================================

dataset = pandas.read_csv("datos_propulsion.csv")

print("\nDATASET ORIGINAL")
print(dataset)

print("\nCANTIDAD DE FILAS Y COLUMNAS")
print(dataset.shape)

print("\nTIPOS DE DATOS")
print(dataset.dtypes)

print("\nDATOS FALTANTES")
print(dataset.isnull().sum())

print("\nFILAS DUPLICADAS")
print(dataset.duplicated().sum())

print("\nRESUMEN ESTADÍSTICO")
print(dataset.describe())

print("\nCANTIDAD DE VALORES DIFERENTES")
print(dataset.nunique())

constant_variables = [
    feature for feature in features
    if dataset[feature].nunique() == 1
]

print("\nENTRADAS CONSTANTES")
print(constant_variables)

print("\nCOMPROBACIÓN DE REDUNDANCIA")
print("Ts y Tp son iguales:", dataset["Ts"].equals(dataset["Tp"]))

# El IQR se utiliza de forma descriptiva. No se eliminan muestras del simulador.
Q1 = dataset[features].quantile(0.25)
Q3 = dataset[features].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

possible_outliers = (
    (dataset[features] < lower_limit)
    | (dataset[features] > upper_limit)
).sum()

print("\nPOSIBLES OUTLIERS SEGÚN IQR")
print(possible_outliers)
print("No se elimina ninguna fila porque son datos generados por el simulador.")

##dataset[targets].hist(bins=25, figsize=(10, 4))
##plt.suptitle("Distribución de las variables objetivo")
##plt.tight_layout()
##plt.show()

# Frecuencia exacta de los valores de las variables objetivo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dataset["kMc"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[0]
)

axes[0].set_title("Frecuencia de cada valor de kMc")
axes[0].set_xlabel("kMc")
axes[0].set_ylabel("Cantidad de muestras")
axes[0].tick_params(axis="x", labelsize=6)

dataset["kMt"].value_counts().sort_index().plot(
    kind="bar",
    ax=axes[1]
)

axes[1].set_title("Frecuencia de cada valor de kMt")
axes[1].set_xlabel("kMt")
axes[1].set_ylabel("Cantidad de muestras")
axes[1].tick_params(axis="x", labelsize=7)

plt.tight_layout()
plt.show()


# =====================================================================
# 2: DIVISIÓN ALEATORIA Y NORMALIZACIÓN
# =====================================================================

# No se utiliza random_state: la selección cambia en cada ejecución.
# Esta única partición se conserva para comparar todas las configuraciones.
testset_original = dataset.sample(frac=test_percentage)
developmentset_original = dataset.drop(testset_original.index)

train_fraction = train_percentage / (
    train_percentage + validation_percentage
)
trainset_original = developmentset_original.sample(frac=train_fraction)
validationset_original = developmentset_original.drop(trainset_original.index)

trainset_original = trainset_original.sample(frac=1).reset_index(drop=True)
validationset_original = validationset_original.sample(frac=1).reset_index(drop=True)
testset_original = testset_original.sample(frac=1).reset_index(drop=True)

print("\nDIVISIÓN ALEATORIA")
print("Entrenamiento:", trainset_original.shape)
print("Validación:", validationset_original.shape)
print("Prueba:", testset_original.shape)
print(
    "Total:",
    len(trainset_original)
    + len(validationset_original)
    + len(testset_original)
)

# Los mínimos y máximos se calculan solamente con entrenamiento.
min_features = trainset_original[features].min(axis=0)
original_difference_features = (
    trainset_original[features].max(axis=0) - min_features
)

min_targets = trainset_original[targets].min(axis=0)
original_difference_targets = (
    trainset_original[targets].max(axis=0) - min_targets
)

print("\nDIFERENCIA MÁXIMO - MÍNIMO DE LAS ENTRADAS")
print(original_difference_features)

print("\nVARIABLES QUE PRODUCIRÍAN DIVISIÓN ENTRE CERO")
print(
    original_difference_features[
        original_difference_features == 0
    ].index.tolist()
)

difference_features = original_difference_features.replace(0, 1)
difference_targets = original_difference_targets.replace(0, 1)

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

print("\nRANGOS NORMALIZADOS DEL ENTRENAMIENTO")
print(trainset[features + targets].agg(["min", "max"]))

print("\nCANTIDAD TOTAL DE NaN DESPUÉS DE NORMALIZAR")
print(
    trainset.isnull().sum().sum()
    + validationset.isnull().sum().sum()
    + testset.isnull().sum().sum()
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
trainset_original[["GTT", "GTn", "mf"]].boxplot(ax=axes[0])
axes[0].set_title("Algunas entradas antes de normalizar")
trainset[["GTT", "GTn", "mf"]].boxplot(ax=axes[1])
axes[1].set_title("Las mismas entradas normalizadas")
plt.tight_layout()
plt.show()


# =====================================================================
# 3: GRID SEARCH
# =====================================================================

total_configurations = (
    len(n_list) * len(units_list) * len(learning_rate_list)
)

results = []
idx = 0

print("\nINICIANDO GRID SEARCH")
print("Cantidad de configuraciones:", total_configurations)
print("Todas utilizan la misma partición en esta ejecución.")
print("El conjunto de prueba todavía no se consulta.")

for n_grid in n_list:
    for units_grid in units_list:
        for learning_rate_grid in learning_rate_list:
            idx += 1
            tf.keras.backend.clear_session()

            print(
                "\nConfiguración", idx, "de", total_configurations,
                "- Capas:", n_grid,
                "Neuronas:", units_grid,
                "LR:", learning_rate_grid
            )

            grid_network = create_network(
                n_grid,
                units_grid,
                learning_rate_grid
            )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            )

            grid_history = grid_network.fit(
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

            validation_evaluation = grid_network.evaluate(
                validationset[features],
                validationset[targets],
                verbose=0
            )

            result = {
                "idx": idx,
                "n": n_grid,
                "units": units_grid,
                "learning_rate": learning_rate_grid,
                "parameters": grid_network.count_params(),
                "epochs_used": len(grid_history.history["loss"]),
                "val_loss": validation_evaluation[0],
                "val_mae": validation_evaluation[1],
                "loss_history": grid_history.history["loss"],
                "val_loss_history": grid_history.history["val_loss"],
                "mae_history": grid_history.history["mae"],
                "val_mae_history": grid_history.history["val_mae"]
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
    "\nLa selección final es manual: observar error, cantidad de",
    "parámetros, épocas y comportamiento de las curvas."
)


grid_rows = 7
grid_columns = 4

fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(16, 23))
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


fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(16, 23))
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


# =====================================================================
# 4: SELECCIÓN MANUAL DE LOS HIPERPARÁMETROS
# =====================================================================

print("\nDIGITE LA CONFIGURACIÓN QUE DESEA UTILIZAR")
print("Revisar primero la tabla y las gráficas del Grid Search.")

n = request_integer(
    "Número de capas ocultas [1, 2, 3]: ",
    n_list
)
units = request_integer(
    "Neuronas por capa [5, 10, 20]: ",
    units_list
)
learning_rate = request_float(
    "Learning rate [0.01, 0.005, 0.001]: ",
    learning_rate_list
)

selected_result = results_table[
    (results_table["n"] == n)
    & (results_table["units"] == units)
    & np.isclose(results_table["learning_rate"], learning_rate)
].iloc[0]

print("\nCONFIGURACIÓN SELECCIONADA")
print("Capas ocultas:", n)
print("Neuronas por capa:", units)
print("Learning rate:", learning_rate)
print("Val_loss obtenido en el Grid Search:", round(selected_result["val_loss"], 7))
print("Val_mae obtenido en el Grid Search:", round(selected_result["val_mae"], 7))
print("Parámetros entrenables:", int(selected_result["parameters"]))
print("Epochs utilizadas en el Grid Search:", int(selected_result["epochs_used"]))


# =====================================================================
# 5: ENTRENAMIENTO Y EVALUACIÓN DEL MODELO FINAL
# =====================================================================

tf.keras.backend.clear_session()
final_network = create_network(n, units, learning_rate)

final_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=patience,
    restore_best_weights=True
)

print("\nENTRENANDO EL MODELO FINAL")
print("Parámetros entrenables:", final_network.count_params())

final_history = final_network.fit(
    x=trainset[features],
    y=trainset[targets],
    validation_data=(validationset[features], validationset[targets]),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[final_early_stopping],
    shuffle=True,
    verbose=1
)

train_evaluation = final_network.evaluate(
    trainset[features], trainset[targets], verbose=0
)
validation_evaluation = final_network.evaluate(
    validationset[features], validationset[targets], verbose=0
)

# El test se consulta una única vez después de escoger la configuración.
test_evaluation = final_network.evaluate(
    testset[features], testset[targets], verbose=0
)

print("\nRESULTADOS NORMALIZADOS")
print("Epochs realizadas:", len(final_history.history["loss"]))
print("MSE entrenamiento:", round(train_evaluation[0], 7))
print("MAE entrenamiento:", round(train_evaluation[1], 7))
print("MSE validación:", round(validation_evaluation[0], 7))
print("MAE validación:", round(validation_evaluation[1], 7))
print("MSE prueba:", round(test_evaluation[0], 7))
print("MAE prueba:", round(test_evaluation[1], 7))

predictions_normalized = final_network.predict(
    testset[features], verbose=0
)
predictions = pandas.DataFrame(
    predictions_normalized,
    columns=targets
)
predictions[targets] = (
    predictions[targets] * difference_targets
) + min_targets

real_values = testset_original[targets].reset_index(drop=True)
errors = real_values - predictions

metrics_test = pandas.DataFrame({
    "MAE": errors.abs().mean()
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


history_df = pandas.DataFrame(final_history.history)
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
    axes[position].scatter(
        real_values[target],
        predictions[target],
        alpha=0.35
    )
    minimum = min(real_values[target].min(), predictions[target].min())
    maximum = max(real_values[target].max(), predictions[target].max())
    axes[position].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="red"
    )
    axes[position].set_xlabel(target + " real")
    axes[position].set_ylabel(target + " estimado")
    axes[position].set_title("Predicción de " + target)
    axes[position].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# =====================================================================
# 6: ANÁLISIS DE SENSIBILIDAD DEL MODELO FINAL
# =====================================================================

perturbation = 0.05
sensitivity_rows = []

print("\nINICIANDO ANÁLISIS LOCAL DE SENSIBILIDAD")
print("Perturbación utilizada: ±5 % del rango normalizado.")

for feature in features:
    if original_difference_features[feature] == 0:
        print("Variable constante, no se analiza:", feature)
        continue

    testset_lower = testset[features].copy()
    testset_upper = testset[features].copy()

    testset_lower[feature] = np.maximum(
        testset_lower[feature] - perturbation, 0
    )
    testset_upper[feature] = np.minimum(
        testset_upper[feature] + perturbation, 1
    )

    predictions_lower = final_network.predict(testset_lower, verbose=0)
    predictions_upper = final_network.predict(testset_upper, verbose=0)

    input_change = (
        testset_upper[feature] - testset_lower[feature]
    ).to_numpy()
    valid_positions = input_change > 0

    local_sensitivity = (
        np.abs(
            predictions_upper[valid_positions]
            - predictions_lower[valid_positions]
        )
        / input_change[valid_positions, np.newaxis]
    )

    normalized_changes = np.abs(
        predictions_upper - predictions_lower
    )
    original_changes = (
        normalized_changes * difference_targets.to_numpy()
    )

    sensitivity_rows.append({
        "Variable": feature,
        "Sensibilidad_kMc": local_sensitivity[:, 0].mean(),
        "Sensibilidad_kMt": local_sensitivity[:, 1].mean(),
        "Cambio_promedio_kMc": original_changes[:, 0].mean(),
        "Cambio_promedio_kMt": original_changes[:, 1].mean(),
        "Sensibilidad_promedio": local_sensitivity.mean()
    })


sensitivity_table = pandas.DataFrame(sensitivity_rows)
sensitivity_table = sensitivity_table.sort_values(
    by="Sensibilidad_promedio",
    ascending=False
).reset_index(drop=True)

print("\nSENSIBILIDAD LOCAL DE LAS ENTRADAS")
print(sensitivity_table.round(7).to_string(index=False))

for target in targets:
    column = "Sensibilidad_" + target
    ordered_table = sensitivity_table.sort_values(
        by=column,
        ascending=False
    )

    print("\nSALIDA", target)
    print("Entrada más significativa:", ordered_table.iloc[0]["Variable"])
    print(
        "Entrada menos significativa no constante:",
        ordered_table.iloc[-1]["Variable"]
    )

print(
    "\nNota: Ts y Tp contienen la misma información. Su sensibilidad",
    "individual debe interpretarse con precaución."
)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

table_kMc = sensitivity_table.sort_values(
    by="Sensibilidad_kMc",
    ascending=True
)
axes[0].barh(table_kMc["Variable"], table_kMc["Sensibilidad_kMc"])
axes[0].set_title("Sensibilidad local de kMc")
axes[0].set_xlabel("Sensibilidad normalizada promedio")
axes[0].set_ylabel("Entrada")
axes[0].grid(axis="x", linestyle="--", alpha=0.5)

table_kMt = sensitivity_table.sort_values(
    by="Sensibilidad_kMt",
    ascending=True
)
axes[1].barh(table_kMt["Variable"], table_kMt["Sensibilidad_kMt"])
axes[1].set_title("Sensibilidad local de kMt")
axes[1].set_xlabel("Sensibilidad normalizada promedio")
axes[1].set_ylabel("Entrada")
axes[1].grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print("\nRED COMPLETADA")
