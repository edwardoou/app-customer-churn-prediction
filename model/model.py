# model/model.py
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Cargar el modelo entrenado
with open('model/model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Cargar el scaler entrenado (debe haberse guardado durante el entrenamiento)
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Función para predecir la deserción de un cliente
def predecir_cliente(data):
    # Suponiendo que 'data' es un DataFrame con las características del cliente
    # Usar el scaler entrenado para transformar los datos
    data_scaled = scaler.transform(data)
    probabilidad = rf_model.predict_proba(data_scaled)[0][1]
    prediccion = rf_model.predict(data_scaled)[0]
    return probabilidad, prediccion
