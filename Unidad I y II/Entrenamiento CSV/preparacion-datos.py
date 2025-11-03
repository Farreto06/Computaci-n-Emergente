import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras

column_names = ['Date','Price', 'Close', 'High', 'Low', 'Open', 'Volume']

# 1. Cargar el dataset (asume que el archivo se llama 'toyota_stock.csv')
data = pd.read_csv(
    'Toyota_stock_data.csv',
    skiprows=3,  # Saltamos Fila 0 (Encabezado), Fila 1 (Ticker), Fila 2 (Date)
    header=None, # No usamos ninguna fila como encabezado (porque la saltamos)
    names=column_names, # Asignamos los nombres correctos
    index_col=0 # Usamos la columna 'Date' (la primera columna) como índice
)

# 2. Diagnóstico (Opcional)
print("Columnas cargadas y limpiadas:", data.columns.tolist())
print("\nPrimeras filas del DataFrame:")
print(data.head())

# data.columns = data.columns.str.strip()

# Usaremos el precio de cierre para la predicción
stock_prices = data['Close'].values.reshape(-1, 1)

# 2. Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_prices)

print("Datos cargados y escalados correctamente. Listo para el modelo.")

# 3. Definir el tamaño del "look back" y crear secuencias
look_back = 60 # Usar los últimos 60 días para predecir el día 61

# Función para crear el dataset de secuencias
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Crear conjuntos de datos (X = Secuencias de entrada, Y = Precio a predecir)
X, y = create_dataset(scaled_data, look_back)

# 4. Reshape para el formato 3D requerido por LSTM
# [muestras, timesteps, características]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 5. Definición del modelo LSTM
model = keras.models.Sequential()

# Primera capa LSTM con 50 neuronas. 'return_sequences=True'
# para pasar la secuencia completa a la siguiente capa LSTM
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(keras.layers.Dropout(0.2)) # 20% de neuronas se apagan aleatoriamente

# Segunda capa LSTM
model.add(keras.layers.LSTM(units=50, return_sequences=False)) # 'return_sequences=False' para la última capa LSTM
model.add(keras.layers.Dropout(0.2))

# Capa de Salida (Dense)
# Una sola neurona, ya que solo predecimos un valor (el precio de cierre)
model.add(keras.layers.Dense(units=1))

# 6. Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Entrenamiento del modelo
# epochs: Número de veces que el modelo verá todos los datos
# batch_size: Número de muestras procesadas antes de actualizar los pesos
history = model.fit(X, y, epochs=50, batch_size=32)

from sklearn.metrics import mean_squared_error

# 8. Predicciones
predicted_prices_scaled = model.predict(X)

# 9. Invertir la transformación para obtener los precios reales
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# 10. Evaluación del modelo (Error Cuadrático Medio, RMSE)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")

import matplotlib.pyplot as plt

# 1. Crear un DataFrame para la gráfica
# NOTA: Los índices de las predicciones corresponden a los datos de entrenamiento
# menos el 'look_back' y el último dato.
# Asegúrate de que 'actual_prices' y 'predicted_prices' tengan la misma longitud.
train_data_length = len(data) - len(actual_prices)

# Crear un array con valores NaN (Not a Number) del mismo tamaño que los datos originales
plot_data = data.filter(['Close'])
train_predictions_plot = np.empty_like(plot_data)
train_predictions_plot[:, :] = np.nan

# Llenar la sección predicha (desde el día 'look_back')
# El inicio de las predicciones es 'look_back', el final es 'len(actual_prices) + look_back'
start_index = look_back
end_index = look_back + len(predicted_prices)
train_predictions_plot[start_index:end_index, :] = predicted_prices

# 2. Configuración de la Gráfica
plt.figure(figsize=(16, 8))
plt.title('Predicción del Precio de Cierre de Acciones de Toyota (LSTM)')
plt.xlabel('Fecha de la Muestra', fontsize=14)
plt.ylabel('Precio de Cierre (USD)', fontsize=14)

# 3. Graficar los datos Reales vs. Predichos

# Precios Reales (Línea Azul)
plt.plot(plot_data.index, plot_data['Close'], label='Precio de Cierre Real', color='blue')

# Precios Predichos por el Modelo (Línea Roja)
# Usamos el array que hemos rellenado para que la línea roja empiece
# exactamente donde empiezan las predicciones (después de los primeros 'look_back' días).
plt.plot(plot_data.index, train_predictions_plot, label='Precio Predicho (Conjunto de Entrenamiento)', color='red')

# 4. Mostrar la Leyenda y la Gráfica
plt.legend(loc='lower right')
plt.grid(True)
plt.show()