# Quick Start Guide - Model Comparison

## Installation

1. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python -c "import numpy, pandas, matplotlib, seaborn, sklearn, xgboost, torch; print('✓ All packages installed successfully!')"
   ```

## Running the Comparison

### One Command
```bash
python model_comparison.py
```

### What to Expect

The script will:
- Take approximately 3-7 minutes to complete (depending on your hardware)
- Print progress updates for each model
- Generate two output files

### Output Files

1. **model_comparison_results.png** - Visual comparison dashboard
2. **model_comparison_summary.csv** - Metrics summary table

## Example Session

```bash
$ python model_comparison.py

================================================================================
WINE QUALITY MODEL COMPARISON
Comparing XGBoost, Deep Neural Network, and K-Means Clustering
================================================================================

================================================================================
LOADING AND PREPROCESSING DATA
================================================================================
Dataset shape: (6362, 16)
...

================================================================================
TRAINING XGBOOST MODEL
================================================================================
[0]     train-logloss:0.67234   test-logloss:0.67845
[20]    train-logloss:0.58123   test-logloss:0.59234
...

================================================================================
TRAINING DEEP NEURAL NETWORK
================================================================================
Epoch 0000 | Train Loss: 0.6534 | Val Loss: 0.6234 | Val Acc: 0.6789 | Val F1: 0.6512
...

================================================================================
TRAINING K-MEANS CLUSTERING
================================================================================
k=2: Silhouette Score = 0.2345
k=3: Silhouette Score = 0.3156
...

================================================================================
MODEL COMPARISON SUMMARY REPORT
================================================================================
          Model  Type      Accuracy  Precision  Recall  F1 Score
       XGBoost  Supervised  0.8234    0.8156    0.7845   0.7998
...

================================================================================
MODEL COMPARISON COMPLETE
================================================================================
```

## Customizing the Comparison

### Quick Parameter Changes

Edit these lines in `model_comparison.py`:

```python
# Line 12: Change random seed
SEED = 42  # Change to any number for different random initialization

# Line 62: Change train/test split
test_size=0.2  # Change to 0.3 for 30% test set

# Line 314: Change DNN epochs
epochs=1000  # Increase to 2000 for longer training
```

## Viewing Results

### 1. Visualization (PNG)
Open `model_comparison_results.png` to see:
- Performance bar charts
- Confusion matrices
- Training curves
- Cluster analysis

### 2. Summary Table (CSV)
Open `model_comparison_summary.csv` in Excel or any spreadsheet software

### 3. Console Output
The terminal shows detailed metrics and recommendations

## Troubleshooting

### Script Fails to Run
```bash
# Check if you're in the right directory
pwd
# Should show: /path/to/5126-Team-Project

# Check if data file exists
ls supervised_model/wine_feature_engineered.csv
```

### Out of Memory Error
- Close other applications
- Or reduce batch size in the script (line 313)

### Slow Performance
- Normal on CPU: 3-7 minutes
- With GPU: 1-2 minutes
- Consider reducing epochs if too slow

## Next Steps

1. Review the generated visualizations
2. Read the summary report recommendations
3. Compare with your teammates' R-based results
4. Decide which model to deploy

## Need Help?

- Full documentation: See `MODEL_COMPARISON_README.md`
- Dataset issues: Check if `wine_feature_engineered.csv` exists
- Package errors: Reinstall with `pip install -r requirements.txt --upgrade`
