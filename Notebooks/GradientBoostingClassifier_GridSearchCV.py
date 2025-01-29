# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:24:02 2024

@author: lazkuenaru
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el CSV
df = pd.read_csv(r'C:\Users\lazkuenaru\Desktop\TFM\pharmaceutical_extraction\src\Model\textos_patologias_linea_de_administracion.csv', 
                 encoding='utf-8-sig', delimiter=';')


df.dropna(inplace=True)
df['label'] = df['administracion'].astype(int)

X = df['textos'] 
y = df['label']

random = 81418

# División de los datos en conjuntos de TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random)

# Vectorización de los textos usando TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Definir la cuadrícula de hiperparámetros para Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300, 50],                        # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1, 0.5],                # Tasa de aprendizaje
    'max_depth': [2, 3, 5, 7, 11],                           # Profundidad máxima de cada árbo
    'min_samples_split': [2, 5, 10, 15, 17],                  # Mínimo de muestras para dividir un nodo
    }

# Configuración con GridSearchCV
gb = GradientBoostingClassifier(random_state=random)
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Mejor modelo
best_gb = grid_search.best_estimator_
print(f"\nMejores hiperparámetros encontrados: {grid_search.best_params_}")

# Evaluación en el conjunto de prueba
y_pred = best_gb.predict(X_test_tfidf)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')  # 'macro' para varias clases
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(6)])

# Mostrar resultados
print("\nResultados del mejor modelo:")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Visualización de resultados
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(['Accuracy', 'Recall'], [accuracy, recall], color=['skyblue', 'salmon'])
plt.title('Mejor modelo - Accuracy y Recall')
plt.ylabel('Score')

# Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[str(i) for i in range(6)], yticklabels=[str(i) for i in range(6)])
plt.title("Matriz de Confusión del Mejor Modelo (Gradient Boosting)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
