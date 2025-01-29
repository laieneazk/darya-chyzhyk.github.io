# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:17:15 2024

@author: lazkuenaru
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

# División de los datos en conjuntos de TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización de los textos usando TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Definir clasificadores con hiperparámetros
classifiers = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=300, C=1.5),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "SVM": SVC(C=1.0, kernel='linear', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
}

# Entrenar y evaluar cada clasificador
results = {}
for name, clf in classifiers.items():
    # Entrenamiento
    clf.fit(X_train_tfidf, y_train)
    
    # Predicción
    y_pred = clf.predict(X_test_tfidf)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')  # 'macro' para varias clases
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(6)])
    
    # Guardar resultados
    results[name] = {
        "Accuracy": accuracy,
        "Recall": recall,
        "Confusion Matrix": conf_matrix,
        "Classification Report": report
    }
    
    # Imprimir resultados
    print(f"\nClasificador: {name}")
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)

# Plotting accuracy and recall
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.bar(results.keys(), [results[clf]["Accuracy"] for clf in results], color=['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'lightblue'])
plt.title('Accuracy por clasificador')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(results.keys(), [results[clf]["Recall"] for clf in results], color=['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'lightblue'])
plt.title('Recall por clasificador')
plt.ylabel('Recall')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Confusion Matrices
for name, clf in classifiers.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(results[name]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", 
                xticklabels=[str(i) for i in range(6)], yticklabels=[str(i) for i in range(6)])
    plt.title(f"Matriz de Confusión: {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
