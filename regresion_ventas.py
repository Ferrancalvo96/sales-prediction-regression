# Predicción de ventas usando regresión lineal
# Dataset de ejemplo: gasto en publicidad (TV, radio, periódico) y ventas

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dataset artificial
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}

df = pd.DataFrame(data)

# Variables
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

# Visualización
plt.scatter(y_test, y_pred)
plt.xlabel("Ventas reales")
plt.ylabel("Ventas predichas")
plt.title("Predicción de ventas")
plt.grid(True)
plt.show()
