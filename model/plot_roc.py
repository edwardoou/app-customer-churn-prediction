import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

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

# Modelos
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)

models = [
    (rf, 'Random Forest'),
    (dt, 'Árbol de Decisión'),
    (lr, 'Regresión Logística'),
    (xgb, 'XGBoost'),
    (ann, 'Red Neuronal')
]

plt.figure(figsize=(10, 6))
for model, label in models:
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC Comparativa')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('static/roc_comparacion.png')
print('Gráfico guardado en static/roc_comparacion.png')
