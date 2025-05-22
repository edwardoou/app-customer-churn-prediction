import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Cargar dataset
df = pd.read_csv('assets/BankChurners.csv')

# Convertir la variable objetivo a binaria
if 'Attrition_Flag' in df.columns:
    df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if 'Attrited' in str(x) else 0)

# Eliminar columnas no necesarias
columns_to_drop = [
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=col)

# Codificar categóricas
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Escalado
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Guardar modelo
with open('model/model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Guardar scaler
with open('model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print('Modelo y scaler entrenados y guardados correctamente.')
