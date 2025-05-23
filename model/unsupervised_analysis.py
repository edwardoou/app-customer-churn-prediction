import pandas as pd
import numpy as np
import matplotlib
# Configurar Matplotlib para usar un backend no interactivo
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA

def perform_unsupervised_analysis(df):
    """
    Realiza análisis no supervisado usando K-means y devuelve solo una tabla simple
    """
    # Crear una copia del dataframe
    df_copy = df.copy()
    
    # Verificar que las columnas necesarias existan
    required_columns = ['Attrition_Flag', 'CLIENT_ID']
    for col in required_columns:
        if col not in df_copy.columns:
            if col == 'CLIENT_ID' and 'CLIENTNUM' in df_copy.columns:
                df_copy['CLIENT_ID'] = df_copy['CLIENTNUM']
            else:
                return f"<p class='text-red-500'>Error: Columna {col} no encontrada en el conjunto de datos.</p>"
    
    # Convertir variables categóricas a numéricas
    cat_cols = df_copy.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Saltarse la columna de ID si es de tipo object
        if col == 'CLIENT_ID':
            continue
        df_copy[col] = pd.factorize(df_copy[col])[0]
    
    # NO SUPERVISADO
    X_unsupervised = df_copy.drop(columns=['Attrition_Flag', 'CLIENT_ID'])
    
    # Verificar si hay columnas problemáticas
    # Eliminar columnas que empiezan con "Naive_Bayes_Classifier" si existen
    cols_to_drop = [col for col in X_unsupervised.columns if col.startswith('Naive_Bayes_Classifier')]
    if cols_to_drop:
        X_unsupervised = X_unsupervised.drop(columns=cols_to_drop)
    
    # Escalado de características
    scaler = StandardScaler()
    X_scaled_unsupervised = scaler.fit_transform(X_unsupervised)

    # Elegir número de clusters
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled_unsupervised)

    # Añadir etiqueta cluster al dataframe
    df_copy['Cluster'] = clusters
    
    # Crear visualización de clusters usando PCA para reducir dimensionalidad
    create_cluster_visualization(X_scaled_unsupervised, clusters, df_copy)

    # Estadísticas por cluster - solo tabla simple
    # Seleccionar columnas numéricas importantes para mostrar en la tabla
    numeric_cols = ['Customer_Age', 'Attrition_Flag']
    for col in numeric_cols:
        if col not in df_copy.columns:
            numeric_cols.remove(col)
    
    # Si no quedan columnas, usar todas las numéricas
    if not numeric_cols:
        numeric_cols = df_copy.select_dtypes(include=['number']).columns[:3]  # Mostrar las primeras 3 columnas numéricas
    
    cluster_stats = df_copy.groupby('Cluster')[numeric_cols].mean()
    
    # Convertir Attrition_Flag a porcentaje para mejor interpretación
    if 'Attrition_Flag' in cluster_stats.columns:
        cluster_stats['Churn_Rate'] = cluster_stats['Attrition_Flag'] * 100
        cluster_stats = cluster_stats.drop(columns=['Attrition_Flag'])
    
    # Diccionario de traducción para los nombres de columnas
    column_translations = {
        'Customer_Age': 'Edad Promedio',
        'Churn_Rate': 'Tasa de Abandono'
    }
    
    # Crear tabla HTML con estilos
    html_table = """
    <table class="min-w-full bg-white border border-gray-300 rounded-lg overflow-hidden">
        <thead class="bg-blue-100">
            <tr>
                <th class="px-4 py-2 text-left text-gray-700 font-semibold">Grupo</th>
    """
    
    # Crear encabezados para cada columna con nombres en español
    for col in cluster_stats.columns:
        # Convertir el nombre de la columna para mostrar en español
        display_name = column_translations.get(col, col.replace('_', ' ').title())
        html_table += f'<th class="px-4 py-2 text-left text-gray-700 font-semibold">{display_name}</th>\n'
    
    html_table += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Filas de datos
    for cluster_id, row in cluster_stats.reset_index().iterrows():
        # Usar números de cluster más amigables (1, 2, 3 en lugar de 0, 1, 2)
        cluster_num = int(row["Cluster"]) + 1
        html_table += '<tr class="border-t border-gray-300 hover:bg-gray-50">\n'
        html_table += f'<td class="px-4 py-2 font-medium">Grupo {cluster_num}</td>\n'
        
        for col in cluster_stats.columns:
            value = row[col]
            # Formatear valores numéricos
            if col == 'Churn_Rate':
                formatted_value = f"{value:.2f}%"
            elif col == 'Customer_Age':
                formatted_value = f"{value:.1f} años"
            elif isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            html_table += f'<td class="px-4 py-2">{formatted_value}</td>\n'
        
        html_table += '</tr>\n'
    
    # Añadir una descripción de los grupos al final de la tabla
    html_table += """
        </tbody>
    </table>
    <div class="mt-4 text-sm text-gray-600">
        <p><strong>Nota:</strong> La tasa de abandono representa el porcentaje de clientes que abandonaron el banco en cada grupo.</p>
        <ul class="mt-2 list-disc pl-5">
            <li><strong>Grupo 1:</strong> Clientes con mayor riesgo de abandono.</li>
            <li><strong>Grupo 2:</strong> Clientes con menor riesgo de abandono.</li>
            <li><strong>Grupo 3:</strong> Clientes con riesgo medio de abandono.</li>
        </ul>
    </div>
    
    <!-- Incluir la imagen generada como parte de la respuesta HTML -->
    <div class="mt-6">
        <h3 class="text-md font-semibold text-gray-700 mb-2">Visualización de Clusters</h3>
        <img src="/static/customer_segments_current.png?t={np.random.randint(10000)}" alt="Segmentación de Clientes" class="w-full rounded shadow border border-gray-200 mt-2">
    </div>
    """
    
    return html_table

def create_cluster_visualization(X_scaled, clusters, df):
    """
    Crea visualización de clusters usando PCA para reducir la dimensionalidad
    y guarda la imagen en la carpeta static
    """
    # Reducir dimensionalidad a 2D usando PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Colores para los clusters
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Dibujar cada cluster con un color diferente
    for i in range(len(np.unique(clusters))):
        plt.scatter(
            X_pca[clusters == i, 0], 
            X_pca[clusters == i, 1],
            c=colors[i], 
            label=f'Grupo {i+1}',
            s=70,
            alpha=0.6,
            edgecolors='w'
        )
    
    # Añadir títulos y leyenda
    plt.title('Segmentación de Clientes', fontsize=15, fontweight='bold')
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.legend(title='Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir anotaciones sobre las tasas de abandono
    if 'Attrition_Flag' in df.columns and 'Cluster' in df.columns:
        churn_rates = df.groupby('Cluster')['Attrition_Flag'].mean() * 100
        
        for i, rate in enumerate(churn_rates):
            # Calcular posiciones para las anotaciones
            x_pos = np.mean(X_pca[clusters == i, 0])
            y_pos = np.mean(X_pca[clusters == i, 1])
            
            # Añadir etiqueta con la tasa de abandono
            plt.annotate(
                f'Tasa de abandono: {rate:.1f}%',
                (x_pos, y_pos),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
            )
    
    plt.tight_layout()
    
    # Asegurar que la carpeta static existe
    static_dir = os.path.join('static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Guardar la imagen
    plt.savefig(os.path.join(static_dir, 'customer_segments_current.png'), dpi=300, bbox_inches='tight')
    plt.close()