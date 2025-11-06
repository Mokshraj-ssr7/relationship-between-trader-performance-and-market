# SETUP INSTRUCTIONS

## Bitcoin Trading Sentiment Analysis Project

---

## 📥 Step 1: Download the Datasets

You need to manually download two CSV files from Google Drive:

### 1. Historical Trading Data
- **Link:** https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing
- **Filename:** `historical_data.csv`
- **Save to:** `c:\Users\lenov\Desktop\datascience\data\historical_data.csv`

### 2. Fear/Greed Index
- **Link:** https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing
- **Filename:** `fear_greed_index.csv`
- **Save to:** `c:\Users\lenov\Desktop\datascience\data\fear_greed_index.csv`

**Instructions:**
1. Click each link above
2. Click "Download" button (top-right corner in Google Drive)
3. Save files to the `data/` directory with the exact names above

---

## 🔧 Step 2: Set Up Python Environment

### Option A: Using venv (Recommended for Windows)

```powershell
# Navigate to project directory
cd c:\Users\lenov\Desktop\datascience

# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using conda

```powershell
# Create conda environment
conda create -n trading-sentiment python=3.10 -y

# Activate environment
conda activate trading-sentiment

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Step 3: Launch Jupyter Notebook

```powershell
# Make sure your environment is activated, then:
jupyter notebook
```

This will open Jupyter in your browser. Navigate to:
- `notebooks/eda.ipynb`

---

## 🧪 Step 4: Run the Analysis

In the Jupyter notebook:
1. Click "Cell" → "Run All" to execute all cells
2. Or run cells one-by-one with `Shift + Enter`

**First cell will check if datasets are present.** If missing, you'll see clear instructions.

---

## ✅ Step 5: Verify Installation

Run this in a notebook cell or Python terminal:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

print("✓ All packages installed successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Plotly version: {plotly.__version__}")
```

---

## 🧪 Step 6: Run Tests (Optional)

```powershell
# Run unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

Coverage report will be saved to `htmlcov/index.html`

---

## 📊 Project Structure Overview

```
datascience/
├── data/                          # Place datasets here
│   ├── historical_data.csv        # ← Download this
│   └── fear_greed_index.csv       # ← Download this
├── notebooks/
│   └── eda.ipynb                  # Main analysis notebook
├── src/
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Data cleaning & merging
│   ├── analysis.py                # Statistical analysis
│   └── features.py                # Feature engineering
├── tests/
│   ├── test_data_loader.py
│   └── test_preprocessing.py
├── deliverables/
│   └── report.md                  # Final report template
├── requirements.txt
├── README.md
└── SETUP.md                       # This file
```

---

## 🔍 What the Analysis Does

1. **Loads Data:** Historical trading data + Fear/Greed sentiment
2. **Cleans & Merges:** Aligns timestamps, aggregates to daily metrics
3. **Analyzes Patterns:**
   - Performance by sentiment (Fear vs Greed)
   - Correlation analysis
   - Lag effects (does past sentiment predict current performance?)
   - Leverage and volume patterns
4. **Statistical Tests:** T-tests, ANOVA, Mann-Whitney U
5. **Visualizations:** 
   - Interactive Plotly charts
   - Correlation heatmaps
   - Time series overlays
   - 3D scatter plots

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```powershell
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Issue: "FileNotFoundError" for datasets

**Solution:**
- Ensure CSVs are in `data/` directory with exact filenames:
  - `historical_data.csv`
  - `fear_greed_index.csv`
- Check file paths in error message

### Issue: Jupyter kernel crashes

**Solution:**
```powershell
# Restart Jupyter
jupyter notebook --notebook-dir=c:\Users\lenov\Desktop\datascience
```

### Issue: Plotly charts not showing

**Solution:**
```powershell
# Install JupyterLab extensions
pip install jupyterlab-plotly
jupyter labextension install jupyterlab-plotly
```

### Issue: Permission error when activating venv

**Solution:**
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📖 Additional Resources

- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Plotly Gallery:** https://plotly.com/python/
- **Seaborn Tutorial:** https://seaborn.pydata.org/tutorial.html
- **Scipy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html

---

## 💡 Quick Start Commands (All-in-One)

```powershell
# 1. Navigate to project
cd c:\Users\lenov\Desktop\datascience

# 2. Create & activate environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Then open `notebooks/eda.ipynb` and run all cells!

---

## 📧 Need Help?

If you encounter issues:
1. Check error messages carefully
2. Review this SETUP.md file
3. Check `README.md` for project overview
4. Consult the troubleshooting section above

---

**Last Updated:** November 6, 2025  
**Python Version Required:** 3.10+  
**Estimated Setup Time:** 5-10 minutes
