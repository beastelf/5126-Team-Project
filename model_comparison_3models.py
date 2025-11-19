"""
Comprehensive 3-Model Comparison Script
Compares XGBoost, Neural Network (MLP), and K-Means Clustering

This script:
1. Trains XGBoost classifier (supervised)
2. Trains Multi-Layer Perceptron / Neural Network (supervised)
3. Trains K-Means clustering (unsupervised)
4. Compares performance metrics
5. Generates comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("WINE QUALITY MODEL COMPARISON - 3 Models")
print("XGBoost vs Neural Network (MLP) vs K-Means Clustering")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load wine dataset and prepare for modeling"""
    print("\n" + "=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    # Load the feature-engineered dataset
    df = pd.read_csv('supervised_model/wine_feature_engineered.csv')
    print(f"Dataset shape: {df.shape}")

    # Prepare features and target
    X = df.drop(columns=['quality', 'wine_type']).values.astype(np.float32)
    y = df['quality'].values

    # Convert to binary classification (quality >= 6 is high quality)
    y_binary = np.where(y >= 6, 1, 0).astype(np.int64)

    print(f"\nClass distribution:")
    print(f"  Low quality (0): {(y_binary == 0).sum()} samples ({(y_binary == 0).sum()/len(y_binary)*100:.1f}%)")
    print(f"  High quality (1): {(y_binary == 1).sum()} samples ({(y_binary == 1).sum()/len(y_binary)*100:.1f}%)")

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
    print(f"Features: {X_train.shape[1]}")

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

    # Set parameters
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
    print("\nTraining XGBoost...")
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

    print(f"\n{'='*60}")
    print(f"XGBoost Results:")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*60}")

    return model, metrics, y_pred, y_pred_proba

# ============================================================================
# 3. NEURAL NETWORK (MLP) MODEL
# ============================================================================

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train and evaluate Multi-Layer Perceptron (Neural Network)"""
    print("\n" + "=" * 80)
    print("TRAINING NEURAL NETWORK (MLP)")
    print("=" * 80)

    # Create MLP classifier with architecture similar to the PyTorch DNN
    print("\nNeural Network Architecture:")
    print("  Input Layer: 14 features")
    print("  Hidden Layer 1: 128 neurons + ReLU")
    print("  Hidden Layer 2: 128 neurons + ReLU")
    print("  Hidden Layer 3: 128 neurons + ReLU")
    print("  Output Layer: 2 classes")
    print("  Regularization: L2 (alpha=0.0005)")

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128, 128),
        activation='relu',
        solver='adam',
        alpha=0.0005,  # L2 regularization
        batch_size=256,
        learning_rate_init=0.00001,
        max_iter=1000,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=False
    )

    # Train model
    print("\nTraining Neural Network...")
    mlp.fit(X_train, y_train)

    print(f"Training completed in {mlp.n_iter_} iterations")
    print(f"Final training loss: {mlp.loss_:.4f}")

    # Make predictions
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'n_iterations': mlp.n_iter_,
        'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else []
    }

    print(f"\n{'='*60}")
    print(f"Neural Network Results:")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*60}")

    return mlp, metrics, y_pred, y_pred_proba

# ============================================================================
# 4. K-MEANS CLUSTERING MODEL
# ============================================================================

def train_kmeans(X_train, X_test, y_train, y_test, k_range=range(2, 9)):
    """Train and evaluate K-Means clustering"""
    print("\n" + "=" * 80)
    print("TRAINING K-MEANS CLUSTERING")
    print("=" * 80)

    # Find optimal k using silhouette score
    print("\nFinding optimal number of clusters...")
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=25, max_iter=100)
        cluster_labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, cluster_labels)
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette Score = {score:.4f}")

    # Select best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal k = {best_k} (Silhouette Score = {max(silhouette_scores):.4f})")

    # Train final model with best k
    print(f"\nTraining K-Means with k={best_k}...")
    kmeans_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=25, max_iter=200)
    train_clusters = kmeans_final.fit_predict(X_train)
    test_clusters = kmeans_final.predict(X_test)

    # Map clusters to quality labels
    cluster_to_label = {}
    for cluster_id in range(best_k):
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            cluster_to_label[cluster_id] = np.bincount(y_train[mask]).argmax()

    print(f"\nCluster to Quality Mapping:")
    for cluster_id, label in cluster_to_label.items():
        count = (train_clusters == cluster_id).sum()
        print(f"  Cluster {cluster_id} -> Quality {'High' if label == 1 else 'Low'} ({count} samples)")

    # Predict labels based on cluster assignments
    y_pred = np.array([cluster_to_label.get(c, 0) for c in test_clusters])

    # Calculate metrics
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
        'silhouette_scores': silhouette_scores
    }

    print(f"\n{'='*60}")
    print(f"K-Means Results:")
    print(f"{'='*60}")
    print(f"  Silhouette Score:      {metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index:   {metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized MI:         {metrics['normalized_mutual_info']:.4f}")
    print(f"  Accuracy (mapped):     {metrics['accuracy']:.4f}")
    print(f"  F1 Score (mapped):     {metrics['f1']:.4f}")
    print(f"{'='*60}")

    return kmeans_final, metrics, y_pred

# ============================================================================
# 5. COMPREHENSIVE VISUALIZATION
# ============================================================================

def plot_3model_comparison(xgb_metrics, nn_metrics, kmeans_metrics):
    """Create comprehensive 3-model comparison visualization"""
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE VISUALIZATION")
    print("=" * 80)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    sns.set_style("whitegrid")

    # 1. Model Performance Comparison (Bar Chart)
    ax1 = plt.subplot(2, 4, 1)
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.25

    xgb_scores = [xgb_metrics[m] for m in metrics_to_compare]
    nn_scores = [nn_metrics[m] for m in metrics_to_compare]
    kmeans_scores = [kmeans_metrics[m] for m in metrics_to_compare]

    bars1 = ax1.bar(x_pos - width, xgb_scores, width, label='XGBoost',
                     alpha=0.85, color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x_pos, nn_scores, width, label='Neural Net',
                     alpha=0.85, color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x_pos + width, kmeans_scores, width, label='K-Means',
                     alpha=0.85, color='#3498db', edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax1.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.capitalize() for m in metrics_to_compare], fontsize=10)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. Confusion Matrix - XGBoost
    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(xgb_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens', ax=ax2, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('XGBoost\nConfusion Matrix', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax2.set_ylabel('True', fontsize=10, fontweight='bold')
    ax2.set_xticklabels(['Low', 'High'], fontsize=9)
    ax2.set_yticklabels(['Low', 'High'], fontsize=9, rotation=0)

    # 3. Confusion Matrix - Neural Network
    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(nn_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Reds', ax=ax3, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax3.set_title('Neural Network\nConfusion Matrix', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax3.set_ylabel('True', fontsize=10, fontweight='bold')
    ax3.set_xticklabels(['Low', 'High'], fontsize=9)
    ax3.set_yticklabels(['Low', 'High'], fontsize=9, rotation=0)

    # 4. Confusion Matrix - K-Means
    ax4 = plt.subplot(2, 4, 4)
    sns.heatmap(kmeans_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax4.set_title('K-Means\nConfusion Matrix', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax4.set_ylabel('True', fontsize=10, fontweight='bold')
    ax4.set_xticklabels(['Low', 'High'], fontsize=9)
    ax4.set_yticklabels(['Low', 'High'], fontsize=9, rotation=0)

    # 5. Neural Network Training Curve
    ax5 = plt.subplot(2, 4, 5)
    if len(nn_metrics['loss_curve']) > 0:
        iterations = range(1, len(nn_metrics['loss_curve']) + 1)
        ax5.plot(iterations, nn_metrics['loss_curve'],
                 color='#e74c3c', linewidth=2.5, marker='o',
                 markersize=3, markevery=max(1, len(iterations)//20))
        ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax5.set_title('Neural Network Training Loss', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3, linestyle='--')

    # 6. K-Means Silhouette Analysis
    ax6 = plt.subplot(2, 4, 6)
    k_values = list(range(2, 2 + len(kmeans_metrics['silhouette_scores'])))
    ax6.plot(k_values, kmeans_metrics['silhouette_scores'],
             marker='o', linewidth=3, markersize=10,
             color='#9b59b6', markeredgecolor='black', markeredgewidth=1.5)
    ax6.axvline(x=kmeans_metrics['best_k'], color='#e74c3c',
                linestyle='--', linewidth=2,
                label=f'Optimal k={kmeans_metrics["best_k"]}')
    ax6.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax6.set_title('K-Means Cluster Selection', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_xticks(k_values)

    # 7. F1 Score Comparison (Bar Chart)
    ax7 = plt.subplot(2, 4, 7)
    models = ['XGBoost', 'Neural\nNetwork', 'K-Means']
    f1_scores = [xgb_metrics['f1'], nn_metrics['f1'], kmeans_metrics['f1']]
    colors_f1 = ['#2ecc71', '#e74c3c', '#3498db']

    bars = ax7.bar(models, f1_scores, color=colors_f1, alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    # Add value labels and winner crown
    max_f1 = max(f1_scores)
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax7.text(bar.get_x() + bar.get_width()/2., score,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        if score == max_f1:
            ax7.text(bar.get_x() + bar.get_width()/2., score + 0.05,
                    '👑 Winner', ha='center', fontsize=10, fontweight='bold')

    ax7.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax7.set_title('F1 Score Comparison\n(Primary Metric)', fontweight='bold', fontsize=12)
    ax7.set_ylim([0, max(f1_scores) * 1.15])
    ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 8. Detailed Metrics Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    table_data = [
        ['Metric', 'XGBoost', 'Neural Net', 'K-Means'],
        ['Accuracy', f"{xgb_metrics['accuracy']:.4f}",
         f"{nn_metrics['accuracy']:.4f}", f"{kmeans_metrics['accuracy']:.4f}"],
        ['Precision', f"{xgb_metrics['precision']:.4f}",
         f"{nn_metrics['precision']:.4f}", f"{kmeans_metrics['precision']:.4f}"],
        ['Recall', f"{xgb_metrics['recall']:.4f}",
         f"{nn_metrics['recall']:.4f}", f"{kmeans_metrics['recall']:.4f}"],
        ['F1 Score', f"{xgb_metrics['f1']:.4f}",
         f"{nn_metrics['f1']:.4f}", f"{kmeans_metrics['f1']:.4f}"],
    ]

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 5):
        for j in range(4):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            if j > 0:  # Highlight best scores
                row_values = [float(table_data[i][1]),
                             float(table_data[i][2]),
                             float(table_data[i][3])]
                if float(table_data[i][j]) == max(row_values):
                    table[(i, j)].set_text_props(weight='bold', color='#27ae60')

    ax8.set_title('Performance Metrics Summary', fontweight='bold', fontsize=12, pad=15)

    plt.suptitle('Wine Quality Prediction: 3-Model Comparison\nXGBoost vs Neural Network vs K-Means',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('model_comparison_3models_results.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("\n✓ Saved visualization: model_comparison_3models_results.png")
    plt.close()

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================

def print_summary_report(xgb_metrics, nn_metrics, kmeans_metrics):
    """Print comprehensive summary report"""
    print("\n" + "=" * 80)
    print("3-MODEL COMPARISON SUMMARY REPORT")
    print("=" * 80)

    # Create summary table
    summary_data = {
        'Model': ['XGBoost', 'Neural Network (MLP)', 'K-Means Clustering'],
        'Type': ['Supervised', 'Supervised', 'Unsupervised'],
        'Accuracy': [
            f"{xgb_metrics['accuracy']:.4f}",
            f"{nn_metrics['accuracy']:.4f}",
            f"{kmeans_metrics['accuracy']:.4f}"
        ],
        'Precision': [
            f"{xgb_metrics['precision']:.4f}",
            f"{nn_metrics['precision']:.4f}",
            f"{kmeans_metrics['precision']:.4f}"
        ],
        'Recall': [
            f"{xgb_metrics['recall']:.4f}",
            f"{nn_metrics['recall']:.4f}",
            f"{kmeans_metrics['recall']:.4f}"
        ],
        'F1 Score': [
            f"{xgb_metrics['f1']:.4f}",
            f"{nn_metrics['f1']:.4f}",
            f"{kmeans_metrics['f1']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Determine winner
    f1_scores = {
        'XGBoost': xgb_metrics['f1'],
        'Neural Network': nn_metrics['f1'],
        'K-Means': kmeans_metrics['f1']
    }
    winner = max(f1_scores, key=f1_scores.get)

    print("\n" + "-" * 80)
    print("WINNER: " + winner)
    print("-" * 80)
    print(f"Best F1 Score: {f1_scores[winner]:.4f}")

    # Detailed recommendations
    print("\n" + "-" * 80)
    print("DETAILED ANALYSIS:")
    print("-" * 80)

    print(f"\n1. XGBoost (Gradient Boosting):")
    print(f"   • F1 Score: {xgb_metrics['f1']:.4f}")
    print(f"   • Accuracy: {xgb_metrics['accuracy']:.4f}")
    print("   • Strengths: Fast training, handles tabular data well, feature importance")

    print(f"\n2. Neural Network (MLP):")
    print(f"   • F1 Score: {nn_metrics['f1']:.4f}")
    print(f"   • Accuracy: {nn_metrics['accuracy']:.4f}")
    print(f"   • Training iterations: {nn_metrics['n_iterations']}")
    print("   • Strengths: Captures non-linear patterns, flexible architecture")

    print(f"\n3. K-Means Clustering:")
    print(f"   • F1 Score: {kmeans_metrics['f1']:.4f}")
    print(f"   • Accuracy: {kmeans_metrics['accuracy']:.4f}")
    print(f"   • Optimal clusters: {kmeans_metrics['best_k']}")
    print(f"   • Silhouette score: {kmeans_metrics['silhouette_score']:.4f}")
    print("   • Strengths: Unsupervised, discovers natural groupings")

    print("\n" + "-" * 80)
    print("RECOMMENDATION FOR PRODUCTION:")
    print("-" * 80)
    print(f"Deploy {winner} for wine quality prediction")
    if winner in ['XGBoost', 'Neural Network']:
        print("Supervised learning shows superior performance for this task")
    print("=" * 80)

    # Save summary
    summary_df.to_csv('model_comparison_3models_summary.csv', index=False)
    print("\n✓ Summary saved to: model_comparison_3models_summary.csv")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Train all three models
    xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    nn_model, nn_metrics, nn_pred, nn_proba = train_neural_network(
        X_train, X_test, y_train, y_test
    )

    kmeans_model, kmeans_metrics, kmeans_pred = train_kmeans(
        X_train, X_test, y_train, y_test
    )

    # Generate visualizations
    plot_3model_comparison(xgb_metrics, nn_metrics, kmeans_metrics)

    # Print summary report
    print_summary_report(xgb_metrics, nn_metrics, kmeans_metrics)

    print("\n" + "=" * 80)
    print("3-MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  ✓ model_comparison_3models_results.png - Comprehensive visualization")
    print("  ✓ model_comparison_3models_summary.csv - Metrics summary table")
    print("\nReady for your presentation!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
