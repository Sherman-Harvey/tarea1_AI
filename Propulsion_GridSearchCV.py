"""Programa completo para estimar kMc y kMt mediante una red neuronal.

1. Lectura y análisis descriptivo.
2. División aleatoria en entrenamiento/desarrollo y prueba.
3. Normalización min-max sin fuga de información.
4. Búsqueda de capas, neuronas y learning rate con GridSearchCV y
   validación cruzada interna.
5. Evaluación del mejor modelo seleccionado automáticamente.
6. Análisis local de sensibilidad.

"""

import tensorflow as tf
import pandas
import numpy as np
from keras import layers, models, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from scikeras.wrappers import KerasRegressor


# Entradas y salidas suministradas en el dataset.
features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

# Porcentajes para la división aleatoria principal.
train_percentage = 0.85
test_percentage = 0.15

# Esta parte de cada ajuste se usa solamente para que Early Stopping
# pueda observar val_loss. No es un tercer conjunto externo.
early_stopping_validation = 0.15

if not np.isclose(train_percentage + test_percentage, 1.0):
    raise ValueError("Los porcentajes de entrenamiento y prueba deben sumar 1.")

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
    """Construye y compila la red con la configuración indicada."""

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
# El test queda separado y GridSearchCV trabaja solamente con trainset.
testset_original = dataset.sample(frac=test_percentage)
trainset_original = dataset.drop(testset_original.index)

trainset_original = trainset_original.sample(frac=1).reset_index(drop=True)
testset_original = testset_original.sample(frac=1).reset_index(drop=True)

print("\nDIVISIÓN ALEATORIA")
print("Entrenamiento/desarrollo:", trainset_original.shape)
print("Prueba:", testset_original.shape)
print(
    "Total:",
    len(trainset_original)
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
testset = testset_original.copy()

for data_normalized, data_original in [
    (trainset, trainset_original),
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
# 3: BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCHCV
# =====================================================================

total_configurations = (
    len(n_list) * len(units_list) * len(learning_rate_list)
)

# Adapto la red de Keras para poder evaluarla con las funciones de sklearn.
regressor = KerasRegressor(
    model=create_network,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=early_stopping_validation,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ],
    shuffle=True,
    verbose=0
)

# Indico los tres hiperparámetros que quiero combinar.
parameter_grid = {
    "model__n": n_list,
    "model__units": units_list,
    "model__learning_rate": learning_rate_list
}

# Divido entrenamiento en cinco partes diferentes para la validación cruzada.
# No fijo random_state para que la distribución cambie en cada ejecución.
cross_validation = KFold(
    n_splits=5,
    shuffle=True
)

# Comparo todas las configuraciones con MSE y MAE.
# El MSE es la métrica que utilizo para escoger y volver a entrenar la mejor.
grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=parameter_grid,
    scoring={
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    },
    refit="mse",
    cv=cross_validation,
    n_jobs=1,
    return_train_score=True,
    verbose=2
)

print("\nINICIANDO GRIDSEARCHCV")
print("Cantidad de configuraciones:", total_configurations)
print("Cantidad de ajustes:", total_configurations * 5)
print("Cada configuración se evalúa mediante validación cruzada de 5 partes.")
print("Early Stopping usa una fracción interna de cada entrenamiento.")
print("Máximo de epochs por ajuste:", epochs)
print("Paciencia de Early Stopping:", patience)
print("El conjunto de prueba todavía no se consulta.")

grid_search.fit(
    trainset[features],
    trainset[targets]
)

# Convierto los puntajes negativos de sklearn en errores positivos para leerlos.
cv_results = grid_search.cv_results_
results_table = pandas.DataFrame({
    "n": cv_results["param_model__n"].astype(int),
    "units": cv_results["param_model__units"].astype(int),
    "learning_rate": cv_results[
        "param_model__learning_rate"
    ].astype(float),
    "MSE_entrenamiento": -cv_results["mean_train_mse"],
    "MSE_CV": -cv_results["mean_test_mse"],
    "Desviacion_MSE_CV": cv_results["std_test_mse"],
    "MAE_CV": -cv_results["mean_test_mae"],
    "Tiempo_promedio": cv_results["mean_fit_time"]
})

results_table = results_table.sort_values(
    by=["MSE_CV", "Tiempo_promedio"],
    ascending=[True, True]
).reset_index(drop=True)

results_table.insert(
    0,
    "Posición",
    np.arange(1, len(results_table) + 1)
)

print("\nRESULTADOS PROMEDIO DE LA VALIDACIÓN CRUZADA")
print(results_table.round(7).to_string(index=False))

best_parameters = grid_search.best_params_
n = int(best_parameters["model__n"])
units = int(best_parameters["model__units"])
learning_rate = float(best_parameters["model__learning_rate"])

print("\nMEJORES HIPERPARÁMETROS SELECCIONADOS AUTOMÁTICAMENTE")
print("Capas ocultas:", n)
print("Neuronas por capa:", units)
print("Learning rate:", learning_rate)
print("MSE promedio de validación cruzada:", round(-grid_search.best_score_, 7))

configuration_labels = (
    "C:" + results_table["n"].astype(str)
    + " N:" + results_table["units"].astype(str)
    + " LR:" + results_table["learning_rate"].astype(str)
)

plt.figure(figsize=(14, 6))
plt.bar(
    configuration_labels,
    results_table["MSE_CV"],
    yerr=results_table["Desviacion_MSE_CV"],
    capsize=3
)
plt.title("MSE promedio de cada configuración en GridSearchCV")
plt.xlabel("Configuración")
plt.ylabel("MSE promedio de validación cruzada")
plt.xticks(rotation=90, fontsize=7)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# =====================================================================
# 4: EVALUACIÓN DEL MODELO SELECCIONADO
# =====================================================================

# Tomo la mejor red, que GridSearchCV volvió a entrenar con todo trainset.
final_network = grid_search.best_estimator_.model_
final_history = grid_search.best_estimator_.history_

print("\nMODELO FINAL SELECCIONADO POR GRIDSEARCHCV")
print("Parámetros entrenables:", final_network.count_params())

train_evaluation = final_network.evaluate(
    trainset[features], trainset[targets], verbose=0
)

# El test se consulta una única vez después de escoger la configuración.
test_evaluation = final_network.evaluate(
    testset[features], testset[targets], verbose=0
)

print("\nRESULTADOS NORMALIZADOS")
print("Epochs realizadas:", len(final_history["loss"]))
print("MSE entrenamiento:", round(train_evaluation[0], 7))
print("MAE entrenamiento:", round(train_evaluation[1], 7))
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


history_df = pandas.DataFrame(final_history)

# Renombro la métrica por si SciKeras la guarda con su nombre completo.
history_df = history_df.rename(columns={
    "mean_absolute_error": "mae",
    "val_mean_absolute_error": "val_mae"
})

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
# 5: ANÁLISIS DE SENSIBILIDAD DEL MODELO FINAL
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
