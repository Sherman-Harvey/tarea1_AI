"""División aleatoria y normalización sin fuga de datos."""

import pandas
import matplotlib.pyplot as plt


features = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf"
]
targets = ["kMc", "kMt"]

train_percentage = 0.70
validation_percentage = 0.15
test_percentage = 0.15

dataset = pandas.read_csv("datos_propulsion.csv")

# No se usa random_state: la selección cambia en cada ejecución.
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
print("Total:", len(trainset_original) + len(validationset_original) + len(testset_original))

# Los valores de normalización se obtienen únicamente del entrenamiento.
min_features = trainset_original[features].min(axis=0)
max_features = trainset_original[features].max(axis=0)
difference_features = max_features - min_features

min_targets = trainset_original[targets].min(axis=0)
max_targets = trainset_original[targets].max(axis=0)
difference_targets = max_targets - min_targets

print("\nDIFERENCIA MÁXIMO - MÍNIMO DE LAS ENTRADAS")
print(difference_features)

print("\nVARIABLES QUE PRODUCIRÍAN DIVISIÓN ENTRE CERO")
print(difference_features[difference_features == 0].index.tolist())

# Una diferencia de 1 evita dividir entre cero. Como valor-mínimo es cero
# para una variable constante, esta quedará normalizada en cero.
difference_features = difference_features.replace(0, 1)
difference_targets = difference_targets.replace(0, 1)

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

