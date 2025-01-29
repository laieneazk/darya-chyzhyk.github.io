# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 30 09:28:50 2024

# @author: lazkuenaru
# """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix,f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset


# Cargar el CSV
df = pd.read_csv(r'C:\Users\lazkuenaru\Desktop\TFM\pharmaceutical_extraction\src\Model\textos_patologias_linea_de_administracion.csv', 
                  encoding='utf-8-sig', 
                  delimiter=';')

# Verifica si hay valores nulos
print("Valores nulos en el DataFrame:")
print(df.isnull().sum())


df['administracion'] = df['administracion'].astype(int)
df.rename(columns={'administracion': 'label'}, inplace=True)

# Dividir los datos en TRAIN y TEST
train_df, test_df = train_test_split(df, test_size=0.3, random_state=81418)

def preprocess_text(text):
    return text.lower()

train_df['textos'] = train_df['textos'].apply(preprocess_text)
test_df['textos'] = test_df['textos'].apply(preprocess_text)

# Convertir los DataFrames a Datasets de Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Cargar el tokenizer y el modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Tokenizar los datos
def tokenize_function(examples):
    return tokenizer(
        examples['textos'], 
        padding = 'max_length', 
        truncation = True, 
        max_length = 512
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy = 'epoch',  # Evaluar al final de cada época
    save_strategy = 'epoch',  # Guardar al final de cada época
    learning_rate = 3e-5,  
    per_device_train_batch_size = 32,  
    per_device_eval_batch_size = 32,
    num_train_epochs = 5,  
    weight_decay = 0.01,
    save_total_limit = 2,  # Limitar el número de modelos guardados
    load_best_model_at_end = True,  # Cargar el mejor modelo al final
    metric_for_best_model = "eval_loss",  # Métrica para determinar el mejor modelo
)

# Definir el data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Definir el entrenador
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_test,
    tokenizer = tokenizer,
    data_collator = data_collator,
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
evaluation_results = trainer.evaluate()

# Imprimir las métricas de evaluación
print("Resultados de la evaluación:")
for key, value in evaluation_results.items():
    print(f"{key}: {value:.4f}")

# Generar predicciones para el conjunto de prueba
predictions = trainer.predict(tokenized_test)

# Obtener las etiquetas predichas
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Acceder a las etiquetas verdaderas correctamente
true_labels = predictions.label_ids

# Mostrar el informe de clasificación
print("\nInforme de clasificación:")
print(classification_report(true_labels, pred_labels, target_names=[str(i) for i in df['label'].unique()]))


accuracy_rf = accuracy_score(true_labels, pred_labels)
recall_rf = recall_score(true_labels, pred_labels, average='macro')
f1_rf = f1_score(true_labels, pred_labels, average='macro')


print("\nAccuracy:")
print(accuracy_rf)
print("\Recall:")
print(recall_rf)
print("\F1:")
print(f1_rf)


# Crear la matriz de confusión
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['label'].unique(), yticklabels=df['label'].unique())
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de confusión")
plt.show()


# from sklearn.model_selection import StratifiedKFold
# from datasets import Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix

# # Configuración de StratifiedKFold
# kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=81418)

# # Inicializar variables para almacenar métricas
# accuracies, recalls, f1_scores = [], [], []

# # Cargar el tokenizer y el modelo
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# # Tokenización
# def tokenize_function(examples):
#     return tokenizer(
#         examples['textos'],
#         padding='max_length',
#         truncation=True,
#         max_length=512
#     )

# # Validación cruzada
# for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_df['textos'], train_df['label']), 1):
#     print(f"\nFold {fold_idx}")
    
#     # Crear conjuntos de entrenamiento y validación
#     fold_train_df = train_df.iloc[train_idx]
#     fold_val_df = train_df.iloc[val_idx]
    
#     # Convertir a Dataset
#     fold_train_dataset = Dataset.from_pandas(fold_train_df)
#     fold_val_dataset = Dataset.from_pandas(fold_val_df)
    
#     # Tokenizar
#     tokenized_train = fold_train_dataset.map(tokenize_function, batched=True)
#     tokenized_val = fold_val_dataset.map(tokenize_function, batched=True)
    
#     # Configuración del entrenador
#     training_args = TrainingArguments(
#         output_dir=f'./results_fold_{fold_idx}',
#         evaluation_strategy='epoch',
#         save_strategy='epoch',
#         learning_rate=3e-5,
#         per_device_train_batch_size=32,
#         per_device_eval_batch_size=32,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_total_limit=1,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss"
#     )
    
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_val,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#     )
    
#     # Entrenar
#     trainer.train()
    
#     # Evaluar
#     predictions = trainer.predict(tokenized_val)
#     pred_labels = np.argmax(predictions.predictions, axis=-1)
#     true_labels = predictions.label_ids
    
#     # Calcular métricas
#     acc = accuracy_score(true_labels, pred_labels)
#     rec = recall_score(true_labels, pred_labels, average='macro')
#     f1 = f1_score(true_labels, pred_labels, average='macro')
    
#     # Almacenar métricas
#     accuracies.append(acc)
#     recalls.append(rec)
#     f1_scores.append(f1)
#     print(fold_idx)
    

# # Promediar las métricas
# avg_accuracy = np.mean(accuracies)
# avg_recall = np.mean(recalls)
# avg_f1 = np.mean(f1_scores)

# # Mostrar resultados promedio
# print("\nResultados Promedio:")
# print(f"Accuracy Promedio: {avg_accuracy:.4f}")
# print(f"Recall Promedio: {avg_recall:.4f}")
# print(f"F1-Score Promedio: {avg_f1:.4f}")
