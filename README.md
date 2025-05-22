App de Predicción de Deserción de Clientes

Esta aplicación web permite predecir la probabilidad de que un cliente de un banco abandone la entidad (“churn”), utilizando un modelo de Machine Learning entrenado con datos históricos reales. 
El usuario ingresa el ID de un cliente y la app muestra:
 - Los datos principales del cliente (edad, género, nivel educativo, estado civil, categoría de ingreso).
 - La probabilidad de deserción calculada por el modelo.
 - Una predicción clara: si el cliente ABANDONA o CONTINÚA con los servicios.

Un gráfico comparativo del rendimiento de varios algoritmos de clasificación (Random Forest, XGBoost, Árbol de Decisión, Regresión Logística, Red Neuronal).

El modelo principal utilizado es Random Forest, seleccionado por su alto rendimiento y robustez. 
La métrica principal de evaluación es el AUC de la curva ROC, ideal para problemas de clasificación desbalanceada.
