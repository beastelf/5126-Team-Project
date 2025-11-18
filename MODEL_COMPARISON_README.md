# Model Comparison Script

## Overview

This script provides a comprehensive comparison between three different machine learning approaches for wine quality classification:

1. **XGBoost** - Gradient boosting classifier (supervised)
2. **Deep Neural Network (DNN)** - Multi-layer perceptron with batch normalization and dropout (supervised)
3. **K-Means Clustering** - Unsupervised clustering with quality label mapping

## Features

### What the Script Does

- Loads and preprocesses the wine quality dataset
- Trains all three models with optimal hyperparameters
- Evaluates performance using multiple metrics
- Generates comprehensive visualizations
- Produces a detailed comparison report

### Metrics Compared

#### Supervised Models (XGBoost & DNN)
- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

#### Unsupervised Model (K-Means)
- **Silhouette Score**: Measure of cluster quality (-1 to 1)
- **Adjusted Rand Index**: Similarity between clustering and true labels
- **Normalized Mutual Information**: Information shared between clusters and labels
- **Mapped Accuracy/F1**: Performance after mapping clusters to quality labels

## Requirements

### Python Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch
```

### Required Packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting framework
- `torch` - Deep learning framework (PyTorch)

## Usage

### Basic Usage

Simply run the script from the repository root:

```bash
python model_comparison.py
```

### Expected Output

The script will:

1. **Load Data**: Reads `supervised_model/wine_feature_engineered.csv`
2. **Train XGBoost**: Trains gradient boosting model with early stopping
3. **Train DNN**: Trains deep neural network with 1000 epochs (or early stops)
4. **Train K-Means**: Finds optimal k using silhouette score (tests k=2 to 8)
5. **Generate Visualizations**: Creates comprehensive comparison plots
6. **Save Results**: Outputs summary files

### Generated Files

After running, you'll get:

1. **model_comparison_results.png** - Comprehensive visualization including:
   - Performance comparison bar chart
   - Confusion matrices for all models
   - DNN training curves (loss, accuracy, F1)
   - K-Means silhouette score curve

2. **model_comparison_summary.csv** - Summary table with all metrics

## Model Details

### XGBoost Configuration
```python
Parameters:
- max_depth: 4
- learning_rate (eta): 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- early_stopping_rounds: 10
```

### DNN Architecture
```python
Layers:
- Input Layer: 14 features
- Hidden Layer 1: 128 units + BatchNorm + ReLU + Dropout(0.2)
- Hidden Layer 2: 128 units + BatchNorm + ReLU + Dropout(0.2)
- Hidden Layer 3: 128 units + BatchNorm + ReLU + Dropout(0.2)
- Output Layer: 2 units (binary classification)

Training:
- Optimizer: Adam (lr=1e-5, weight_decay=5e-4)
- Loss: CrossEntropyLoss
- Early Stopping: Patience=100 epochs
```

### K-Means Configuration
```python
Parameters:
- k_range: 2 to 8 clusters
- Selection: Maximum silhouette score
- n_init: 25 random initializations
- max_iter: 200 iterations
```

## Understanding the Results

### Interpreting the Visualization

1. **Performance Comparison (Top Left)**
   - Higher bars indicate better performance
   - Compare all three models across four metrics
   - Look for consistent performance across metrics

2. **Confusion Matrices (Top Middle & Right)**
   - Diagonal elements: Correct predictions
   - Off-diagonal: Misclassifications
   - Balanced matrix indicates good generalization

3. **DNN Training Curves (Bottom Left & Middle)**
   - Loss should decrease over epochs
   - Train/Val curves should be close (no overfitting)
   - F1 score should stabilize

4. **K-Means Silhouette (Bottom Right)**
   - Peak indicates optimal number of clusters
   - Higher silhouette score = better-defined clusters

### Recommendations Section

The script automatically provides recommendations:

- **Best Supervised Model**: Based on F1 score
- **Clustering Quality**: Assessment of unsupervised approach
- **Production Deployment**: Guidance on which model to use

## Example Output

```
================================================================================
MODEL COMPARISON SUMMARY REPORT
================================================================================

          Model  Type      Accuracy  Precision  Recall  F1 Score
       XGBoost  Supervised  0.8234    0.8156    0.7845   0.7998
Deep Neural Network  Supervised  0.8187    0.8023    0.7912   0.7967
K-Means Clustering  Unsupervised  0.6543    0.6234    0.6891   0.6545

--------------------------------------------------------------------------------
K-Means Clustering Specific Metrics:
  Silhouette Score:        0.3245
  Adjusted Rand Index:     0.2156
  Normalized Mutual Info:  0.1987
  Optimal k:               3

--------------------------------------------------------------------------------
RECOMMENDATIONS:
--------------------------------------------------------------------------------

1. Best Supervised Model: XGBoost
   - F1 Score: 0.7998
   - XGBoost performs better, likely due to:
     * Better handling of tabular data
     * Built-in feature importance
     * Faster training time

2. Unsupervised Learning (K-Means):
   - K-Means shows limited performance (F1=0.6545)
   - Quality labels may not align well with feature-based clusters
   - Optimal number of clusters: 3
   - Silhouette score: 0.3245

3. Overall Recommendation:
   - Use XGBoost for production deployment
   - Model shows strong predictive performance
```

## Customization

### Adjusting Model Parameters

You can modify the following in the script:

```python
# Data split
test_size = 0.2  # Change train/test split ratio

# XGBoost
params['max_depth'] = 6  # Increase tree depth

# DNN
epochs = 2000  # More training epochs
batch_size = 512  # Larger batches
lr = 1e-4  # Different learning rate

# K-Means
k_range = range(2, 12)  # Test more cluster numbers
```

### Adding More Models

To add additional models to the comparison:

1. Create a training function following the pattern
2. Return metrics in the same format
3. Add to the visualization function
4. Update the summary report

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: wine_feature_engineered.csv`
- **Solution**: Make sure you're running the script from the repository root

**Issue**: `CUDA out of memory` (if using GPU)
- **Solution**: Reduce batch_size in the DNN training function

**Issue**: DNN training is very slow
- **Solution**: Reduce epochs or enable GPU acceleration

**Issue**: Different results each run
- **Solution**: The SEED is set to 42, but some randomness may persist. Results should be very similar.

## Performance Notes

- **XGBoost**: Fastest training, typically 10-30 seconds
- **DNN**: Slowest training, typically 2-5 minutes (CPU) or 30-60 seconds (GPU)
- **K-Means**: Fast training, typically 5-10 seconds

## References

- XGBoost: Chen & Guestrin (2016)
- Deep Learning: Goodfellow et al. (2016)
- K-Means: MacQueen (1967)
- Silhouette Score: Rousseeuw (1987)

## Author

Created for the 5126 Team Project - Wine Quality Analysis

## License

MIT License
