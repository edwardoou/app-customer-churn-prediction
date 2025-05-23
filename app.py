
from flask import Flask, render_template, request
from model.model import predecir_cliente
import pandas as pd
import os
from model.unsupervised_analysis import perform_unsupervised_analysis

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    client_id = request.form['client_id']  # Obtiene el ID del cliente enviado desde el formulario
    # Define la ruta al archivo CSV con los datos originales
    csv_path = os.path.join('assets', 'BankChurners.csv')
    df = pd.read_csv(csv_path)  # Lee el archivo CSV y lo carga en un DataFrame
    # Busca la fila correspondiente al cliente usando su ID
    cliente_row = df[df['CLIENTNUM'] == int(client_id)]
    if cliente_row.empty:
        # Si no se encuentra el cliente, muestra una alerta en la interfaz
        return render_template('index.html', cliente_no_existe=True)

    # Extrae los datos descriptivos del cliente para mostrarlos en la interfaz
    cliente_info = cliente_row.iloc[0]
    datos_cliente = {
        'ID': cliente_info['CLIENTNUM'],  # ID único del cliente
        'Edad': cliente_info['Customer_Age'],  # Edad del cliente
        'Género': cliente_info['Gender'],  # Género del cliente
        'Nivel educativo': cliente_info['Education_Level'],  # Nivel educativo
        'Estado civil': cliente_info['Marital_Status'],  # Estado civil
        'Categoría de ingreso': cliente_info['Income_Category']  # Categoría de ingreso
    }

    # Preprocesa los datos del cliente igual que en el entrenamiento del modelo
    if 'Attrition_Flag' in cliente_row.columns:
        # Convierte la columna de deserción a valores numéricos (1 si desertó, 0 si no)
        cliente_row['Attrition_Flag'] = cliente_row['Attrition_Flag'].apply(lambda x: 1 if 'Attrited' in str(x) else 0)
    columns_to_drop = [
        'CLIENTNUM',  # Elimina el ID del cliente (no es usado para predecir)
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    for col in columns_to_drop:
        if col in cliente_row.columns:
            cliente_row = cliente_row.drop(columns=col)  # Elimina columnas innecesarias
    cat_cols = cliente_row.select_dtypes(include=['object']).columns  # Selecciona columnas categóricas
    for col in cat_cols:
        cliente_row[col] = pd.factorize(cliente_row[col])[0]  # Convierte categorías a números
    if 'Attrition_Flag' in cliente_row.columns:
        cliente_row = cliente_row.drop(columns=['Attrition_Flag'])  # Elimina la columna objetivo si existe
    # Realiza la predicción usando el modelo cargado
    probabilidad, prediccion = predecir_cliente(cliente_row)
    # Renderiza la plantilla mostrando la probabilidad y predicción, junto con los datos del cliente
    return render_template('index.html', probabilidad=probabilidad*100, prediccion="ABANDONA Nuestros servicios" if prediccion == 1 else "CONTINUA con nuestros servicios", datos_cliente=datos_cliente)

import os

@app.route('/get_cluster_analysis')
def get_cluster_analysis():
    try:
        # Leer el CSV
        csv_path = os.path.join('assets', 'BankChurners.csv')
        df = pd.read_csv(csv_path)
        
        # Renombrar la columna para que coincida con lo esperado por la función
        if 'CLIENTNUM' in df.columns and 'CLIENT_ID' not in df.columns:
            df = df.rename(columns={'CLIENTNUM': 'CLIENT_ID'})
        
        # Realizar análisis no supervisado para obtener la tabla HTML
        cluster_stats_html = perform_unsupervised_analysis(df)
        
        # Devolver solo la tabla HTML
        return cluster_stats_html
    except Exception as e:
        # Manejo de errores
        return f"<p class='text-red-500'>Error al generar el análisis: {str(e)}</p>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
