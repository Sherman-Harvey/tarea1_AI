"""Convertir el archivo original data.txt a CSV."""

import pandas


columnas = [
    "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48",
    "T1", "T2", "P48", "P1", "P2", "Pexh", "TIC", "mf",
    "kMc", "kMt"
]

# El archivo original no tiene encabezados y separa los valores con espacios.
dataset = pandas.read_csv(
    "data.txt",
    sep=r"\s+", #indica que las columnas están separadas por uno o más espacios en blanco. Incluye espacios y tabulaciones.
    header=None, #indica que la primera fila del archivo no contiene nombres de columnas, sino datos.
    names=columnas #asigna a las columnas los nombres guardados previamente en la variable columnas
)

print("\nPRIMERAS FILAS INTERPRETADAS")
print(dataset.head())

print("\nCANTIDAD DE FILAS Y COLUMNAS")
print(dataset.shape)

if dataset.shape[1] != 18:
    raise ValueError("La lectura no produjo las 18 columnas esperadas.")

if dataset.isnull().any().any():
    raise ValueError("La conversión produjo datos faltantes; revise data.txt.")

dataset.to_csv("datos_propulsion.csv", index=False)

print("\nSe creó datos_propulsion.csv correctamente.")


