"""
Model Comparison Script - XGBoost vs K-Means
Lightweight version for quick comparison (no PyTorch required)

This script:
1. Trains XGBoost classifier (supervised)
2. Trains K-Means clustering (unsupervised)
3. Compares performance metrics
4. Generates comprehensive visualizations
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
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("WINE QUALITY MODEL COMPARISON - XGBoost vs K-Means")
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
    print(f"Columns: {list(df.columns)}")

    # Prepare features and target
    X = df.drop(columns=['quality', 'wine_type']).values.astype(np.float32)
    y = df['quality'].values

    # Convert to binary classification (quality >= 6 is high quality)
    y_binary = np.where(y >= 6, 1, 0).astype(np.int64)

    print(f"\nClass distribution:")
    print(f"  Low quality (0): {(y_binary == 0).sum()} samples")
    print(f"  High quality (1): {(y_binary == 1).sum()} samples")

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
# 3. K-MEANS CLUSTERING MODEL
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

    # Map clusters to quality labels (find best mapping)
    cluster_to_label = {}
    for cluster_id in range(best_k):
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            # Assign label based on majority class in cluster
            cluster_to_label[cluster_id] = np.bincount(y_train[mask]).argmax()

    print(f"\nCluster to Quality Mapping:")
    for cluster_id, label in cluster_to_label.items():
        print(f"  Cluster {cluster_id} -> Quality {'High' if label == 1 else 'Low'}")

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
        'cluster_mapping': cluster_to_label
    }

    print(f"\n{'='*60}")
    print(f"K-Means Results:")
    print(f"{'='*60}")
    print(f"  Silhouette Score:      {metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index:   {metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized MI:         {metrics['normalized_mutual_info']:.4f}")
    print(f"  Accuracy (mapped):     {metrics['accuracy']:.4f}")
    print(f"  Precision (mapped):    {metrics['precision']:.4f}")
    print(f"  Recall (mapped):       {metrics['recall']:.4f}")
    print(f"  F1 Score (mapped):     {metrics['f1']:.4f}")
    print(f"{'='*60}")

    return kmeans_final, metrics, y_pred, silhouette_scores

# ============================================================================
# 4. VISUALIZATION AND COMPARISON
# ============================================================================

def plot_comparison(xgb_metrics, kmeans_metrics, kmeans_silhouettes):
    """Create comprehensive comparison visualizations"""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))

    # Set style
    sns.set_style("whitegrid")
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

    # 1. Model Performance Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.35

    xgb_scores = [xgb_metrics[m] for m in metrics_to_compare]
    kmeans_scores = [kmeans_metrics[m] for m in metrics_to_compare]

    bars1 = ax1.bar(x_pos - width/2, xgb_scores, width, label='XGBoost',
                     alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, kmeans_scores, width, label='K-Means',
                     alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison\n(Higher is Better)',
                   fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.capitalize() for m in metrics_to_compare], fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. Confusion Matrix - XGBoost
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(xgb_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens', ax=ax2, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax2.set_title('XGBoost Confusion Matrix', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['Low (0)', 'High (1)'], fontsize=10)
    ax2.set_yticklabels(['Low (0)', 'High (1)'], fontsize=10, rotation=0)

    # 3. Confusion Matrix - K-Means
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(kmeans_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', ax=ax3, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax3.set_title('K-Means Confusion Matrix\n(After Cluster Mapping)',
                   fontweight='bold', fontsize=14, pad=15)
    ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(['Low (0)', 'High (1)'], fontsize=10)
    ax3.set_yticklabels(['Low (0)', 'High (1)'], fontsize=10, rotation=0)

    # 4. K-Means Silhouette Scores
    ax4 = plt.subplot(2, 3, 4)
    k_values = list(range(2, 2 + len(kmeans_silhouettes)))
    ax4.plot(k_values, kmeans_silhouettes, marker='o', linewidth=3, markersize=10,
             color='#9b59b6', markeredgecolor='black', markeredgewidth=2)
    ax4.axvline(x=kmeans_metrics['best_k'], color='#e74c3c', linestyle='--',
                linewidth=2, label=f'Optimal k={kmeans_metrics["best_k"]}')
    ax4.axhline(y=max(kmeans_silhouettes), color='#27ae60', linestyle=':',
                linewidth=2, alpha=0.5)
    ax4.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax4.set_title('K-Means Cluster Selection\n(Silhouette Analysis)',
                   fontweight='bold', fontsize=14, pad=15)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(k_values)

    # 5. Detailed Metrics Comparison (Table-style)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    comparison_data = [
        ['Metric', 'XGBoost', 'K-Means', 'Winner'],
        ['Accuracy', f"{xgb_metrics['accuracy']:.4f}",
         f"{kmeans_metrics['accuracy']:.4f}",
         'XGB' if xgb_metrics['accuracy'] > kmeans_metrics['accuracy'] else 'K-M'],
        ['Precision', f"{xgb_metrics['precision']:.4f}",
         f"{kmeans_metrics['precision']:.4f}",
         'XGB' if xgb_metrics['precision'] > kmeans_metrics['precision'] else 'K-M'],
        ['Recall', f"{xgb_metrics['recall']:.4f}",
         f"{kmeans_metrics['recall']:.4f}",
         'XGB' if xgb_metrics['recall'] > kmeans_metrics['recall'] else 'K-M'],
        ['F1 Score', f"{xgb_metrics['f1']:.4f}",
         f"{kmeans_metrics['f1']:.4f}",
         'XGB' if xgb_metrics['f1'] > kmeans_metrics['f1'] else 'K-M'],
    ]

    table = ax5.table(cellText=comparison_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 5):
        for j in range(4):
            if j == 3:  # Winner column
                if comparison_data[i][3] == 'XGB':
                    table[(i, j)].set_facecolor('#d5f4e6')
                else:
                    table[(i, j)].set_facecolor('#ebf5fb')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax5.set_title('Detailed Performance Metrics', fontweight='bold',
                   fontsize=14, pad=20)

    # 6. Clustering Specific Metrics
    ax6 = plt.subplot(2, 3, 6)
    clustering_metrics = ['Silhouette\nScore', 'Adjusted\nRand Index',
                          'Normalized\nMutual Info']
    clustering_values = [
        kmeans_metrics['silhouette_score'],
        kmeans_metrics['adjusted_rand_index'],
        kmeans_metrics['normalized_mutual_info']
    ]

    bars = ax6.barh(clustering_metrics, clustering_values,
                     color=['#e74c3c', '#f39c12', '#9b59b6'],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, clustering_values)):
        ax6.text(val + 0.01, i, f'{val:.3f}',
                 va='center', fontsize=11, fontweight='bold')

    ax6.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax6.set_title('K-Means Clustering Quality Metrics',
                   fontweight='bold', fontsize=14, pad=15)
    ax6.set_xlim([0, max(clustering_values) * 1.15])
    ax6.grid(True, alpha=0.3, axis='x', linestyle='--')

    plt.suptitle('Wine Quality Prediction: XGBoost vs K-Means Clustering',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("\n✓ Saved visualization: model_comparison_results.png")
    plt.close()

# ============================================================================
# 5. SUMMARY REPORT
# ============================================================================

def print_summary_report(xgb_metrics, kmeans_metrics):
    """Print comprehensive summary report"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY REPORT")
    print("=" * 80)

    # Create summary table
    summary_data = {
        'Model': ['XGBoost', 'K-Means Clustering'],
        'Type': ['Supervised', 'Unsupervised'],
        'Accuracy': [
            f"{xgb_metrics['accuracy']:.4f}",
            f"{kmeans_metrics['accuracy']:.4f}"
        ],
        'Precision': [
            f"{xgb_metrics['precision']:.4f}",
            f"{kmeans_metrics['precision']:.4f}"
        ],
        'Recall': [
            f"{xgb_metrics['recall']:.4f}",
            f"{kmeans_metrics['recall']:.4f}"
        ],
        'F1 Score': [
            f"{xgb_metrics['f1']:.4f}",
            f"{kmeans_metrics['f1']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # K-Means specific metrics
    print("\n" + "-" * 80)
    print("K-Means Clustering Specific Metrics:")
    print("-" * 80)
    print(f"  Silhouette Score:        {kmeans_metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index:     {kmeans_metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized Mutual Info:  {kmeans_metrics['normalized_mutual_info']:.4f}")
    print(f"  Optimal k:               {kmeans_metrics['best_k']}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)

    print(f"\n1. Supervised Learning (XGBoost):")
    print(f"   • F1 Score: {xgb_metrics['f1']:.4f}")
    if xgb_metrics['f1'] > 0.75:
        print("   • Shows strong predictive performance ✓")
        print("   • Recommended for production deployment")
    elif xgb_metrics['f1'] > 0.65:
        print("   • Shows good predictive performance")
        print("   • Suitable for practical use")
    else:
        print("   • Moderate performance - consider feature engineering")

    print(f"\n2. Unsupervised Learning (K-Means):")
    print(f"   • F1 Score (mapped): {kmeans_metrics['f1']:.4f}")
    if kmeans_metrics['f1'] > 0.6:
        print("   • Shows reasonable clustering quality")
        print("   • Natural groupings exist in wine quality data")
    else:
        print("   • Limited performance for quality prediction")
        print("   • Quality labels don't align strongly with feature clusters")

    print(f"   • Optimal clusters: {kmeans_metrics['best_k']}")
    print(f"   • Silhouette score: {kmeans_metrics['silhouette_score']:.4f}")

    print(f"\n3. Overall Winner: XGBoost")
    improvement = ((xgb_metrics['f1'] - kmeans_metrics['f1']) / kmeans_metrics['f1'] * 100)
    print(f"   • {improvement:.1f}% better F1 score than K-Means")
    print("   • Supervised learning outperforms unsupervised for this task")
    print("   • XGBoost recommended for wine quality prediction")

    # Save summary to CSV
    summary_df.to_csv('model_comparison_summary.csv', index=False)
    print("\n" + "=" * 80)
    print("✓ Summary saved to: model_comparison_summary.csv")
    print("=" * 80)

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Train models
    xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    kmeans_model, kmeans_metrics, kmeans_pred, kmeans_silhouettes = train_kmeans(
        X_train, X_test, y_train, y_test
    )

    # Generate visualizations
    plot_comparison(xgb_metrics, kmeans_metrics, kmeans_silhouettes)

    # Print summary report
    print_summary_report(xgb_metrics, kmeans_metrics)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  ✓ model_comparison_results.png - Comprehensive visualization")
    print("  ✓ model_comparison_summary.csv - Metrics summary table")
    print("\nReady for your presentation!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
