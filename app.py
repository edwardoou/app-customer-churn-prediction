
from flask import Flask, render_template, request
from model.model import predecir_cliente
import pandas as pd
import os

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    client_id = request.form['client_id']
    # Leer el CSV original
    csv_path = os.path.join('assets', 'BankChurners.csv')
    df = pd.read_csv(csv_path)
    # Buscar la fila del cliente
    cliente_row = df[df['CLIENTNUM'] == int(client_id)]
    if cliente_row.empty:
        return render_template('index.html', probabilidad=None, prediccion=f"No se encontró el cliente con ID {client_id}")

    # Extraer datos descriptivos
    cliente_info = cliente_row.iloc[0]
    datos_cliente = {
        'ID': cliente_info['CLIENTNUM'],
        'Edad': cliente_info['Customer_Age'],
        'Género': cliente_info['Gender'],
        'Nivel educativo': cliente_info['Education_Level'],
        'Estado civil': cliente_info['Marital_Status'],
        'Categoría de ingreso': cliente_info['Income_Category']
    }

    # Procesar igual que en el entrenamiento
    if 'Attrition_Flag' in cliente_row.columns:
        cliente_row['Attrition_Flag'] = cliente_row['Attrition_Flag'].apply(lambda x: 1 if 'Attrited' in str(x) else 0)
    columns_to_drop = [
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    for col in columns_to_drop:
        if col in cliente_row.columns:
            cliente_row = cliente_row.drop(columns=col)
    cat_cols = cliente_row.select_dtypes(include=['object']).columns
    for col in cat_cols:
        cliente_row[col] = pd.factorize(cliente_row[col])[0]
    if 'Attrition_Flag' in cliente_row.columns:
        cliente_row = cliente_row.drop(columns=['Attrition_Flag'])
    probabilidad, prediccion = predecir_cliente(cliente_row)
    return render_template('index.html', probabilidad=probabilidad*100, prediccion="ABANDONA Nuestros servicios" if prediccion == 1 else "CONTINUA con nuestros servicios", datos_cliente=datos_cliente)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
