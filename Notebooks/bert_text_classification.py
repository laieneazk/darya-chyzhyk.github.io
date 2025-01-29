# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:11:35 2024

@author: lazkuenaru
"""

# Importar las bibliotecas necesarias
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Paso 1: Configuración de datos
df = pd.read_csv(r'C:\Users\lazkuenaru\Desktop\TFM\pharmaceutical_extraction\src\Model\textos_patologias_linea_de_administracion.csv', 
                 encoding='utf-8-sig', delimiter=';')

# Paso 2: Preparación del Dataset y DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Dividir los datos y crear DataLoaders para entrenamiento y prueba
texts = df['textos'].tolist()
labels = df['administracion'].tolist()

dataset = TextDataset(texts, labels)

# Dividir el dataset en 80% entrenamiento y 20% prueba
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Paso 3: Configuración del modelo de BERT para clasificación en 6 clases
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Paso 4: Configuración del optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Paso 5: Función de entrenamiento del modelo
def train(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Loss: {epoch_loss / len(dataloader)}")

# Paso 6: Entrenamiento
train(model, train_loader, optimizer, epochs=30)

# Paso 7: Evaluación del modelo
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            _, preds = torch.max(outputs.logits, dim=1)
            all_labels.extend(batch['labels'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Cálculo de métricas
    conf_matrix_test = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Matriz de Confusión para el conjunto de prueba
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[str(i) for i in range(6)], yticklabels=[str(i) for i in range(6)])
    plt.title("Matriz de Confusión del Modelo de Ensemble en Conjunto de Prueba")
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Real")
    plt.show()

    print("Confusion Matrix:\n", conf_matrix_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Llamada a la función de evaluación en el conjunto de prueba
evaluate(model, test_loader)


"""
epochs = 10

Loss: 0.4016532480716705
 Confusion Matrix:
 [[8 0 2 0 0 0]
 [0 6 3 0 0 0]
 [4 0 5 0 0 0]
 [1 0 1 2 1 0]
 [0 0 1 0 2 0]
 [0 0 0 3 0 1]]
Accuracy: 0.6000
Precision: 0.6726
Recall: 0.6000
F1 Score: 0.6011

"""


"""
epochs = 20

Loss: 0.046954667288810016
Confusion Matrix:
 [[8 2 1 0 0 0]
 [0 9 1 0 0 1]
 [2 2 5 0 0 0]
 [0 0 0 3 1 0]
 [0 0 1 0 1 0]
 [1 0 0 1 0 1]]
Accuracy: 0.6750
Precision: 0.6685
Recall: 0.6750
F1 Score: 0.6686

"""


"""
epochs = 30

Loss: 0.016726394603028893
 Confusion Matrix:
 [[ 8  0  2  0  1  0]
 [ 1 10  0  0  0  0]
 [ 0  2  3  0  2  0]
 [ 0  0  1  1  2  2]
 [ 0  1  0  0  0  0]
 [ 1  0  1  0  0  2]]
Accuracy: 0.6000
Precision: 0.7065
Recall: 0.6000
F1 Score: 0.6065
"""
