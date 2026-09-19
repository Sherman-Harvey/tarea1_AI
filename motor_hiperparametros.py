import tensorflow as tf
import pandas as pd
from keras import layers, models, Input
import matplotlib.pyplot as plt
import itertools
import os
import random
import numpy as np

#asegura resultados reproducibles fijando la cantidad de decimales y la semilla aleatoria para todas las librerías utilizadas
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42) # Fuerza el determinismo global en Keras

# Para garantizar operaciones consistentes en CPU
tf.config.experimental.enable_op_determinism()


# Se lee el archivo data.txt 
df_txt = pd.read_csv("data.txt", sep="\s+", header=None)


# Asignamos los nombres de las 16 entradas y 2 salidas
columnas = [
    "Lever position", "Ship speed", "GT  torque", "GT rpm", "GG rpm", 
    "Starboard Torque", "Port Torque", "HP Turbine exit temp", 
    "GT Compressor inlet temp", "GT Compressor outlet temp", 
    "HP Turbine exit pressure", "GT Compressor inlet pressure", 
    "GT Compressor outlet pressure", "GT exhaust gas pressure", 
    "Turbine Injection Control", "Fuel flow", 
    "GT Compressor decay", "GT Turbine decay"
]

#targets = ["Compressor decay", "Turbine decay"]
df_txt.columns = columnas

# Se guarda la conversión y se vuelve a leer como CSV
df_txt.to_csv("datos_navales.csv", index=False)
dataset = pd.read_csv("datos_navales.csv").dropna()

#Se eliminan las columnas constantes que causan división por cero
dataset = dataset.drop(columns=["GT Compressor inlet temp", "GT Compressor inlet pressure"])

print(dataset.isnull().sum())

# Usamos 'dataset.columns' para que extraiga los nombres de la tabla ya limpia.
# Toma desde el inicio hasta la posición 13 (las 14 variables de entrada válidas).
features = dataset.columns[:14] 

# Definimos manualmente las 2 salidas objetivo.
targets = ["GT Compressor decay", "GT Turbine decay"]

# Lectura y normalización de datos
max_val = dataset.max(axis=0) # Se obtiene el máximo de cada columna
min_val = dataset.min(axis=0)# Se obtiene el mínimo de cada columna
difference = max_val - min_val # Se obtiene la diferencia de los dos
new_dataset = (dataset - min_val) / difference # Y se utiliza para normalizarlas
print(new_dataset.max(axis=0))

#Se extrae el 80% de los datos totales exclusivamente para el
#estudio de hiperparámetros
study_dataset = new_dataset.sample(frac=0.70, random_state=42)

# Se aparta el 20% restante para pruebas futuras definitivas
final_test_dataset = new_dataset.drop(study_dataset.index)

# Hiperparámetros
capas_list = [1, 2, 3]         # Capas ocultas
units_list = [5, 3, 2]         # Neuronas por capa
lr_list = [0.01, 0.001, 0.5, 0.0005]   # Tasa de aprendizaje
batch_size_fijo = 500          # <- Batch size fijo

combinaciones = list(itertools.product(capas_list, units_list, lr_list))
resultados = []

# FUNCIONES DE PARADA (STOPPING CRITERIA)

# Función 1: Criterio de parada por X iteraciones máximas recomendadas
max_epochs = 100  #valor relativamente arbitrario(ajustar según la necesidad)


# Función 2: Early Stopping Dinámico por separación de curvas (Overfitting)
#función que detecta el sobreentrenamiento y detiene el entrenamiento de la red neuronal, callback propio
class DeteccionSobreentrenamiento(tf.keras.callbacks.Callback):
    #función que define paciencia y el mejor hiperaparametro
    def __init__(self, patience=5, restore_best_weights=True):
        super().__init__()
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        # Variables internas para rastrear el progreso geométrico de las curvas
        self.wait = 0
        self.best_gap = float('inf')
        self.best_weights = None

        self.best_epoch = 1  # <--iteración óptima

    #función que define los valores de ambas curvas al terminar cada iteración
    def on_epoch_end(self, epoch, logs=None):

        # Se extraen los valores actuales de ambas curvas al terminar la iteración
        current_val_loss = logs.get('val_loss')
        current_loss = logs.get('loss')
        
        # CÁLCULO DINÁMICO

        # Mide la separación matemática entre la validación(val_loss) y el entrenamiento(loss)
        current_gap = current_val_loss - current_loss
        
        # Si la separación se reduce (las curvas convergen), el modelo está generalizando bien
        if current_gap < self.best_gap:
            self.best_gap = current_gap
            self.wait = 0  # Reinicia el contador de paciencia
            self.best_epoch = epoch + 1  # <--Guarda el número real de la mejor época
            
            # Se guardan los pesos de la red neuronal en este punto óptimo
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                
        else:
            # Si el gap aumenta, la curva val_loss comienza a separarse de loss (Sobreentrenamiento)
            self.wait += 1
            
            # Si la separación continúa durante el límite de paciencia, abortamos
            if self.wait >= self.patience:
                print(f"\n[Parada] Convergencia óptima alcanzada en la iteración {self.best_epoch}.")
                self.model.stop_training = True  # Criterio de parada activado
                
                # Se restaura la red al punto de máxima convergencia
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)



# Ciclo for para evaluar cada combinación
for idx, (capas, units_list, lr) in enumerate(combinaciones):


    # Instanciación de la nueva función (Esta es la variable que inyectas en callbacks=[early_stopper])
    # Al crearlo aquí adentro, su memoria (best_weights, wait, best_gap) 
    # se reinicia a cero para cada nueva red neuronal.
    early_stopper = DeteccionSobreentrenamiento(patience=5, restore_best_weights=True)

    # División en entrenamiento y validación
    # ATENCIÓN: Ahora aplicamos el hiperparámetro (80% o 55%) SOLO sobre el study_dataset
    trainset = study_dataset.sample(frac=0.8, random_state=42) ## Se extraen datos al azar del conjunto para el entrenamiento
    testset = study_dataset.drop(trainset.index) #= # Y se le quitan esos mismos
                                                    # al dataset para crear los datos de prueba
    #idx es la desvición estándar de la iteración(gradiente de error), que se utiliza para identificar cada combinación de hiperparámetros.
    # Inicialización del modelo
    
    # Inicialización del modelo
    network = models.Sequential()

    # Capa de entrada: 16 variables simultáneamente
    network.add(Input(shape=(14,)))

    # Capas ocultas dinámicas: Aquí SÍ usamos el hiperparámetro iterativo (5, 3 o 2)
    for _ in range(capas):
        network.add(layers.Dense(
            units=units_list,
            activation="relu"   # no hay datos negativos, por lo que no se necesita la función de activación tangente hiperbólica
            )
        )

    # Capa de salida: FIJA EN 2 SALIDAS
    network.add(layers.Dense(
        units=2,            # <- CORRECCIÓN: Fijo en 2 variables (Compressor decay y Turbine decay)
        activation=None
        )
    )

    # Compilación
    network.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        loss="mse",
        metrics=["mae"] # error absoluto promedio sobre entrenamiento
    )

    # Entrenamiento (verbose=0 para limpiar la terminal)

    print(f"Entrenando config {idx+1}: Capas={capas}, Neuronas={units_list}, LR={lr}")
    history = network.fit(
        x=trainset[features],
        y=trainset[targets],
        validation_data=(
            testset[features],
            testset[targets]
        ),
        batch_size=batch_size_fijo,
        epochs=max_epochs,            # <- Implementación Función 1
        callbacks=[early_stopper],    # <- Early stopper
        verbose=0
    )
    iter_optima = early_stopper.best_epoch

    # Almacenamiento de resultados
    resultados.append({
        'idx': idx + 1,
        'label': f"C:{capas}|N:{units_list}|LR:{lr}",
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'optima': iter_optima 
    })
# Graficación de resultados en matrices 4x4
plots_per_image = 16
for i in range(0, len(resultados), plots_per_image):
    chunk = resultados[i:i + plots_per_image]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f'Gráficas {i+1} a {i+len(chunk)}')
    axes = axes.flatten()
    
    for j, res in enumerate(chunk):
        ax = axes[j]
        ax.plot(res['loss'], label='Entrenamiento (loss)')
        ax.plot(res['val_loss'], label='Validación (val_loss)')


        ax.set_title(f"#{res['idx']} {res['label']} | Óptima: Ep {res['optima']}", fontsize=8)  
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("MSE", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Eliminar recuadros vacíos si la matriz no se llena
    for j in range(len(chunk), 16):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()