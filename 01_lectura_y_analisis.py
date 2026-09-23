"""Lectura, comprensión y análisis descriptivo del dataset."""

import pandas
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

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

variables_constantes = []
for variable in features:
    if dataset[variable].nunique() == 1:
        variables_constantes.append(variable)

print("\nENTRADAS CONSTANTES")
print(variables_constantes)

print("\nCOMPROBACIÓN DE REDUNDANCIA")
print("Ts y Tp son iguales:", dataset["Ts"].equals(dataset["Tp"]))

# IQR descriptivo. No se eliminan datos porque representan condiciones
# generadas intencionalmente por el simulador de la planta.
Q1 = dataset[features].quantile(0.25)
Q3 = dataset[features].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

posibles_outliers = (
    (dataset[features] < limite_inferior)
    | (dataset[features] > limite_superior)
).sum()

print("\nPOSIBLES OUTLIERS SEGÚN IQR")
print(posibles_outliers)
print("No se elimina ninguna fila en esta etapa.")

##dataset[targets].hist(bins=25, figsize=(10, 4))
##plt.suptitle("Distribución de las dos variables objetivo")
##plt.tight_layout()
##plt.show()

#Con el codigo comentado anteriormente no se mostraban bien los valores
#Por lo que se modifico por el siguiente codigo
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
