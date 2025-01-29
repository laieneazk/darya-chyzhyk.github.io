# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:38:07 2024

@author: lazkuenaru
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight


"""
Modelos de Redes Neuronales con Arquitectura Multietiqueta

Arquitectura: red neuronal con una capa de salida de 4 nodos, cada uno con una
              activación sigmoide.A diferencia de softmax, la activación sigmoide
              permite que cada nodo prediga de forma independiente la probabilidad 
              de pertenecer a cada clase. Esto es ideal en un problema multietiqueta,
              ya que cada clase es tratada de manera individual en lugar de hacer una
              predicción exclusiva para una sola clase.
"""

# Cargar el DataFrame
df = pd.read_csv(r'C:\Users\lazkuenaru\Desktop\TFM\pharmaceutical_extraction\src\Model\textos_patologias_linea_de_administracion.csv', 
                 encoding='utf-8-sig', delimiter=';')

# Crear una nueva columna transformando la columna 'administracion'
df['administracion_multi'] = df['administracion'].apply(
    lambda x: [1, 2] if x == 4 else ([2, 3] if x == 5 else [x])
)

# Características y etiquetas
X = df['textos']  # Características
y = df['administracion_multi']  # Usar la nueva columna multiclase

# Convertir texto a características usando TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X).toarray()  # Convertir a matriz 2D

# Codificación de las etiquetas usando MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)  # Conversión a formato binario

# Calcular pesos de clase
unique_classes = np.unique(y_encoded.flatten())
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=unique_classes, 
    y=y_encoded.flatten()
)
class_weights_dict = {i: class_weights[i] for i in unique_classes}

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.3, random_state=42)

# Definir el modelo
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))  # Primera capa oculta
model.add(Dropout(0.3))  # Capa de Dropout para prevenir el sobreajuste
model.add(Dense(64, activation='sigmoid'))  # Segunda capa oculta
model.add(Dropout(0.3))  # Otra capa de Dropout
model.add(Dense(16, activation='relu'))  # Tercera capa oculta
model.add(Dropout(0.3))  # Capa de Dropout
model.add(Dense(y_encoded.shape[1], activation='sigmoid'))  # Capa de salida con Sigmoid

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con pesos de clase
model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

# Evaluación en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # Umbral para convertir las probabilidades en clases

# Imprimir las clases para verificar su contenido
print("Clases codificadas:", mlb.classes_)
print("Predicciones:", y_pred_classes)
print("Clases reales:", y_test)

# Reporte de clasificación
# Convertir las clases codificadas a un formato adecuado
print(classification_report(y_test, y_pred_classes, target_names=[str(c) for c in mlb.classes_]))

# Matriz de confusión para visualización
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_classes.argmax(axis=1))
print("Matriz de confusión:\n", conf_matrix)
