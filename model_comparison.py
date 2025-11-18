"""
Comprehensive Model Comparison Script
Compares XGBoost, Deep Neural Network, and K-Means Clustering on Wine Quality Dataset

This script:
1. Trains XGBoost classifier (supervised)
2. Trains Deep Neural Network (supervised)
3. Trains K-Means clustering (unsupervised)
4. Compares performance metrics
5. Generates comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import KMeans
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load wine dataset and prepare for modeling"""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    # Load the feature-engineered dataset
    df = pd.read_csv('supervised_model/wine_feature_engineered.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Prepare features and target
    X = df.drop(columns=['quality', 'wine_type']).values.astype(np.float32)
    y = df['quality'].values

    # Convert to binary classification (quality >= 6 is high quality)
    y_binary = np.where(y >= 6, 1, 0).astype(np.int64)

    print(f"\nClass distribution:")
    print(f"  Low quality (0): {(y_binary == 0).sum()}")
    print(f"  High quality (1): {(y_binary == 1).sum()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, stratify=y_binary, random_state=SEED
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# 2. XGBOOST MODEL
# ============================================================================

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost classifier"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters (matching the R code)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': SEED,
        'eval_metric': 'logloss'
    }

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=20
    )

    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"\nXGBoost Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    return model, metrics, y_pred, y_pred_proba

# ============================================================================
# 3. DEEP NEURAL NETWORK MODEL
# ============================================================================

class DNNClassifier(nn.Module):
    """Deep Neural Network for binary classification"""
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=2):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

def train_dnn(X_train, X_test, y_train, y_test, epochs=1000, batch_size=256, lr=1e-5):
    """Train and evaluate Deep Neural Network"""
    print("\n" + "=" * 80)
    print("TRAINING DEEP NEURAL NETWORK")
    print("=" * 80)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = DNNClassifier(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'val_f1': []
    }

    # Early stopping variables
    best_val_acc = 0
    patience_counter = 0
    patience = 100

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

                all_preds.extend(outputs.argmax(1).numpy())
                all_labels.extend(y_batch.numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float()
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(1).numpy()
        y_pred_proba = F.softmax(outputs, dim=1)[:, 1].numpy()

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"\nDNN Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    return model, metrics, y_pred, y_pred_proba, history

# ============================================================================
# 4. K-MEANS CLUSTERING MODEL
# ============================================================================

def train_kmeans(X_train, X_test, y_train, y_test, k_range=range(2, 9)):
    """Train and evaluate K-Means clustering"""
    print("\n" + "=" * 80)
    print("TRAINING K-MEANS CLUSTERING")
    print("=" * 80)

    # Find optimal k using silhouette score
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=25, max_iter=100)
        cluster_labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, cluster_labels)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.4f}")

    # Select best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nBest k = {best_k} (Silhouette Score = {max(silhouette_scores):.4f})")

    # Train final model with best k
    kmeans_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=25, max_iter=200)
    train_clusters = kmeans_final.fit_predict(X_train)
    test_clusters = kmeans_final.predict(X_test)

    # Map clusters to quality labels (find best mapping)
    cluster_to_label = {}
    for cluster_id in range(best_k):
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            # Assign label based on majority class in cluster
            cluster_to_label[cluster_id] = np.bincount(y_train[mask]).argmax()

    # Predict labels based on cluster assignments
    y_pred = np.array([cluster_to_label.get(c, 0) for c in test_clusters])

    # Calculate metrics (for unsupervised, we use different metrics)
    metrics = {
        'silhouette_score': silhouette_score(X_test, test_clusters),
        'adjusted_rand_index': adjusted_rand_score(y_test, y_pred),
        'normalized_mutual_info': normalized_mutual_info_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'best_k': best_k,
        'cluster_mapping': cluster_to_label
    }

    print(f"\nK-Means Results:")
    print(f"  Silhouette Score:  {metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized MI:       {metrics['normalized_mutual_info']:.4f}")
    print(f"  Accuracy (mapped):   {metrics['accuracy']:.4f}")
    print(f"  F1 Score (mapped):   {metrics['f1']:.4f}")

    return kmeans_final, metrics, y_pred, silhouette_scores

# ============================================================================
# 5. VISUALIZATION AND COMPARISON
# ============================================================================

def plot_comparison(xgb_metrics, dnn_metrics, kmeans_metrics, dnn_history, kmeans_silhouettes):
    """Create comprehensive comparison visualizations"""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Model Performance Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.25

    xgb_scores = [xgb_metrics[m] for m in metrics_to_compare]
    dnn_scores = [dnn_metrics[m] for m in metrics_to_compare]
    kmeans_scores = [kmeans_metrics[m] for m in metrics_to_compare]

    ax1.bar(x_pos - width, xgb_scores, width, label='XGBoost', alpha=0.8)
    ax1.bar(x_pos, dnn_scores, width, label='DNN', alpha=0.8)
    ax1.bar(x_pos + width, kmeans_scores, width, label='K-Means', alpha=0.8)

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.capitalize() for m in metrics_to_compare])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # 2. Confusion Matrix - XGBoost
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(xgb_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('XGBoost Confusion Matrix', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    # 3. Confusion Matrix - DNN
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(dnn_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax3)
    ax3.set_title('DNN Confusion Matrix', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')

    # 4. DNN Training History - Loss
    ax4 = plt.subplot(2, 3, 4)
    epochs_range = range(1, len(dnn_history['train_loss']) + 1)
    ax4.plot(epochs_range, dnn_history['train_loss'], label='Train Loss', alpha=0.8)
    ax4.plot(epochs_range, dnn_history['val_loss'], label='Val Loss', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('DNN Training & Validation Loss', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. DNN Training History - Accuracy & F1
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs_range, dnn_history['train_acc'], label='Train Accuracy', alpha=0.8)
    ax5.plot(epochs_range, dnn_history['val_acc'], label='Val Accuracy', alpha=0.8)
    ax5.plot(epochs_range, dnn_history['val_f1'], label='Val F1', alpha=0.8)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Score')
    ax5.set_title('DNN Training Metrics', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. K-Means Silhouette Scores
    ax6 = plt.subplot(2, 3, 6)
    k_values = range(2, 2 + len(kmeans_silhouettes))
    ax6.plot(k_values, kmeans_silhouettes, marker='o', linewidth=2, markersize=8)
    ax6.axvline(x=kmeans_metrics['best_k'], color='r', linestyle='--',
                label=f'Best k={kmeans_metrics["best_k"]}')
    ax6.set_xlabel('Number of Clusters (k)')
    ax6.set_ylabel('Silhouette Score')
    ax6.set_title('K-Means Cluster Selection', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: model_comparison_results.png")
    plt.close()

def print_summary_report(xgb_metrics, dnn_metrics, kmeans_metrics):
    """Print comprehensive summary report"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY REPORT")
    print("=" * 80)

    # Create summary table
    summary_data = {
        'Model': ['XGBoost', 'Deep Neural Network', 'K-Means Clustering'],
        'Type': ['Supervised', 'Supervised', 'Unsupervised'],
        'Accuracy': [
            f"{xgb_metrics['accuracy']:.4f}",
            f"{dnn_metrics['accuracy']:.4f}",
            f"{kmeans_metrics['accuracy']:.4f}"
        ],
        'Precision': [
            f"{xgb_metrics['precision']:.4f}",
            f"{dnn_metrics['precision']:.4f}",
            f"{kmeans_metrics['precision']:.4f}"
        ],
        'Recall': [
            f"{xgb_metrics['recall']:.4f}",
            f"{dnn_metrics['recall']:.4f}",
            f"{kmeans_metrics['recall']:.4f}"
        ],
        'F1 Score': [
            f"{xgb_metrics['f1']:.4f}",
            f"{dnn_metrics['f1']:.4f}",
            f"{kmeans_metrics['f1']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Additional clustering metrics
    print("\n" + "-" * 80)
    print("K-Means Clustering Specific Metrics:")
    print(f"  Silhouette Score:        {kmeans_metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index:     {kmeans_metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized Mutual Info:  {kmeans_metrics['normalized_mutual_info']:.4f}")
    print(f"  Optimal k:               {kmeans_metrics['best_k']}")

    # Determine best model
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)

    supervised_models = {
        'XGBoost': xgb_metrics['f1'],
        'DNN': dnn_metrics['f1']
    }
    best_supervised = max(supervised_models, key=supervised_models.get)

    print(f"\n1. Best Supervised Model: {best_supervised}")
    print(f"   - F1 Score: {supervised_models[best_supervised]:.4f}")

    if xgb_metrics['f1'] > dnn_metrics['f1']:
        print("   - XGBoost performs better, likely due to:")
        print("     * Better handling of tabular data")
        print("     * Built-in feature importance")
        print("     * Faster training time")
    else:
        print("   - DNN performs better, likely due to:")
        print("     * Better capture of non-linear relationships")
        print("     * More complex feature interactions")

    print("\n2. Unsupervised Learning (K-Means):")
    if kmeans_metrics['f1'] > 0.6:
        print(f"   - K-Means shows reasonable performance (F1={kmeans_metrics['f1']:.4f})")
        print("   - Suggests natural clustering in wine quality data")
    else:
        print(f"   - K-Means shows limited performance (F1={kmeans_metrics['f1']:.4f})")
        print("   - Quality labels may not align well with feature-based clusters")

    print(f"   - Optimal number of clusters: {kmeans_metrics['best_k']}")
    print(f"   - Silhouette score: {kmeans_metrics['silhouette_score']:.4f}")

    print("\n3. Overall Recommendation:")
    if supervised_models[best_supervised] > 0.75:
        print(f"   - Use {best_supervised} for production deployment")
        print("   - Model shows strong predictive performance")
    else:
        print("   - Consider further feature engineering or ensemble methods")
        print("   - Current models show moderate performance")

    # Save summary to file
    summary_df.to_csv('model_comparison_summary.csv', index=False)
    print("\n" + "=" * 80)
    print("Summary saved to: model_comparison_summary.csv")
    print("=" * 80)

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("WINE QUALITY MODEL COMPARISON")
    print("Comparing XGBoost, Deep Neural Network, and K-Means Clustering")
    print("=" * 80)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Train models
    xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    dnn_model, dnn_metrics, dnn_pred, dnn_proba, dnn_history = train_dnn(
        X_train, X_test, y_train, y_test, epochs=1000
    )

    kmeans_model, kmeans_metrics, kmeans_pred, kmeans_silhouettes = train_kmeans(
        X_train, X_test, y_train, y_test
    )

    # Generate visualizations
    plot_comparison(xgb_metrics, dnn_metrics, kmeans_metrics, dnn_history, kmeans_silhouettes)

    # Print summary report
    print_summary_report(xgb_metrics, dnn_metrics, kmeans_metrics)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - model_comparison_results.png (visualization)")
    print("  - model_comparison_summary.csv (summary table)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
