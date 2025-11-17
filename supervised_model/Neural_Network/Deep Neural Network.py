"""
DNN-based binary classifier for wine quality prediction.

This module loads the combined red/white wine dataset, preprocesses
features, converts the quality score into a binary target, and trains
a multilayer perceptron (MLP) using PyTorch. The network includes
Batch Normalization and Dropout to improve training stability and
reduce overfitting.

Key parameters:
    - nn_in:  number of input features
    - nn_out: number of output classes (binary)
    - nn_neural: width of hidden layers
    - epochs: training iterations
    - lr: learning rate for Adam optimizer
    - batch_train / batch_val: batch sizes for training and validation
    - seed: random seed for reproducibility

The script performs:
    - dataset loading and normalization
    - train/validation split
    - model training and validation
    - final evaluation with accuracy, precision, recall, F1, and confusion matrix
    - visualization of loss, accuracy, and F1 curves
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    f1_score, precision_score, recall_score,
)
import pandas as pd


nn_in = 14
nn_out = 2
epochs = 5000
nn_neural = 128
batch_train = 256
batch_val = 32
lr = 1e-5
seed = 141
torch.manual_seed(seed)
np.random.seed(seed)

origin_resample=pd.read_csv("wine_feature_engineered.csv")
df=pd.DataFrame(origin_resample)
df=df.drop(columns=["wine_type"])

X = df.drop(columns=["quality"]).to_numpy(dtype=np.float32)
y = df['quality'].to_numpy(dtype=np.int64)
# Convert to binary target (<=6 : 0 , >6 : 1)
y = np.where(y >= 6, 1, 0).astype(np.int64)
# Split before normalization to avoid data leakage
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed
)

# Standardization based on training statistics
mean_train = x_train_np.mean(axis=0, keepdims=True)
std_train  = x_train_np.std(axis=0, keepdims=True) + 1e-8
x_train_np = (x_train_np - mean_train) / std_train
x_test_np  = (x_test_np  - mean_train) / std_train

print("Train class counts:", np.bincount(y_train_np))
print("Test  class counts:", np.bincount(y_test_np))
print("X shape:", X.shape)

# Create PyTorch dataloaders
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np)),
    batch_size=batch_train, shuffle=True
)
val_loader   = DataLoader(
    TensorDataset(torch.from_numpy(x_test_np), torch.from_numpy(y_test_np)),
    batch_size=batch_val, shuffle=False
)

# MLP model with BN + Dropout
model = nn.Sequential(
    nn.Linear(nn_in, nn_neural),
    nn.BatchNorm1d(nn_neural),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(nn_neural, nn_neural),
    nn.BatchNorm1d(nn_neural),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(nn_neural, nn_neural),
    nn.BatchNorm1d(nn_neural),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(nn_neural, nn_out),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# History storage for training curves
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'val_f1':[]}
#Early stop
last_acc=0
early_stop=0
stop_epoch=0
# Training loop
for ep in range(epochs):
    #if the accuracy does not increase in 300 epoches, training stop
    if early_stop>=300:
        stop_epoch=ep-1
        break
    model.train()
    train_running_loss, train_correct, n_train = 0.0, 0, 0

    for x, yb in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        n_train += x.size(0)

    train_loss = train_running_loss / max(1, n_train)
    train_acc  = train_correct / max(1, n_train)

    model.eval()
    val_running_loss, val_correct, n_val = 0.0, 0, 0
    all_labels, all_preds = [], []
    prob_list = []

    # Validation loop (no gradient)
    with torch.no_grad():
        for x, yb in val_loader:
            logits = model(x)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)

            val_running_loss += loss.item() * x.size(0)
            val_correct += (probs.argmax(1) == yb).sum().item()
            n_val += x.size(0)

            all_labels.extend(yb.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            prob_list.append(probs.cpu())

    val_loss = val_running_loss / max(1, n_val)
    val_acc  = val_correct / max(1, n_val)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = torch.cat(prob_list, dim=0).numpy()
    if(last_acc<=val_acc):
        last_acc = val_acc
        early_stop = 0
    else:
        early_stop+=1

    f1_macro = f1_score(all_labels, all_preds, average='macro')

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(f1_macro)

    print(f"Epoch {ep:03d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} "
          f"| TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | Macro-F1={f1_macro:.4f}")

# Final evaluation on full test set
model.eval()
x_val = torch.from_numpy(x_test_np)
y_val = torch.from_numpy(y_test_np)

with torch.no_grad():
    logits = model(x_val)
    loss = criterion(logits, y_val)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1)
    Accuracy = (pred == y_val).sum().item() / y_val.shape[0]

cm = confusion_matrix(y_val, pred)
precision = precision_score(y_val, pred)
recall = recall_score(y_val, pred)
f1 = f1_score(y_val, pred)

print("The Validation Accuracy is: ", Accuracy)
print("The Confusion Matrix is: ", cm,"\n")
print("The Precision is: ", precision)
print("The Recall is: ", recall)
print("The F1 is: ", f1)

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Loss curve
epochs_range = range(1, len(history['train_loss']) + 1)
plt.figure(figsize=(8,5))
plt.plot(epochs_range, history['train_loss'], label='Train Loss')
plt.plot(epochs_range, history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy curve
plt.figure(figsize=(8,5))
plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# F1 curve
plt.figure(figsize=(8,5))
plt.plot(epochs_range, history['val_f1'], label='Val Macro-F1', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.title('Validation F1 Score')
plt.legend()
plt.grid(True)
plt.show()

