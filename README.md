# Energy Bill Prediction - Machine Learning Project

## ADDIS ABABA INSTITUTE OF TECHNOLOGY
### Department of Software Engineering

---

## Project Title
**Estimating Monthly Electricity Costs from Appliance Usage Hours**

---

## Team Members

| Name | ID |
|------|-----|
| 1. Kassahun Belachew | ATE/8400/14 |
| 2. Yeabsira Samuel | ATE/9305/14 |
| 3. Natnael Nigatu | ATE/7495/14 |
| 4. Tsegaab Alemu | ATE/8814/14 |

---

## Submission Details
- **Date:** November 22, 2025
- **Submitted to:** Mr. Bisrat

---

## Project Overview
This project implements a Linear Regression model to predict monthly electricity bills for Ethiopian households based on appliance usage patterns.

## Key Results
- **R² Score:** 0.9894 (98.94% variance explained)
- **MAE:** 22.87 ETB
- **RMSE:** 28.68 ETB

## Features Used
- House size (sqm)
- Number of occupants
- Season (hot/cool)
- AC, Fridge, Lights, Fans, Washing Machine, TV usage hours

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```


# ⚡ Energy Bill Predictor - Ethiopia

**Supervised Learning Project: Linear Regression**

A machine learning system that predicts monthly electricity bills based on household characteristics and appliance usage patterns, using Linear Regression with Gradient Descent optimization.

**Currency**: Ethiopian Birr (ETB)

---

## 📚 Project Overview

### Course Information
- **Course**: Supervised Learning - Linear Regression
- **Algorithm**: Linear Regression with Gradient Descent & Feature Scaling
- **Date**: November 2024
- **Objective**: Demonstrate complete understanding of Linear Regression theory, mathematics, and practical implementation

### Problem Statement
Predict monthly electricity bills for Ethiopian households based on:
- House characteristics (size, occupants, season)
- Appliance usage patterns (hours per day)

### Why This Project?
Understanding electricity consumption patterns helps:
- 💰 Households budget better and save money
- 🌍 Reduce energy waste through identification of high-cost appliances
- 📊 Identify that **AC usage is the dominant cost driver** (182.07 ETB/hour)
- 🎓 Demonstrate complete Linear Regression mastery with mathematical rigor

---

## 🎯 Features

### Input Features (9 total)

**House Characteristics:**
1. `house_size_sqm` - House size (50-200 sqm training range, accepts up to 1000 sqm with warnings)
2. `num_occupants` - Number of people (1-6)
3. `season` - Cool (0) or Hot (1)

**Appliance Daily Usage (hours):**
4. `ac` - Air Conditioner (0-24 hours) - **HIGHEST IMPACT (coefficient: 182.07)**
5. `fridge` - Refrigerator (0-24 hours, typically 23-24)
6. `lights` - Lights (0-24 hours)
7. `fans` - Fans (0-24 hours)
8. `washing_machine` - Washing Machine (0-24 hours)
9. `tv` - Television (0-24 hours)

**Note**: All appliance inputs now accept 0-24 hours for flexibility, with intelligent warnings for values outside typical training range.

### Output
- `bill` - Predicted monthly electricity bill in **Ethiopian Birr (ETB)**

---

## 🏆 Model Performance - EXCEEDED EXPECTATIONS!

### Achieved Metrics ✅
- **R² Score**: **0.9894** (98.94% variance explained) 🎉
- **MAE**: **22.87 ETB** (Mean Absolute Error) - **EXCEEDED GOAL BY 75%!**
- **RMSE**: **28.68 ETB** (Root Mean Squared Error)
- **Improvement**: **89.9%** better than baseline (predicting mean)
- **Training Time**: **< 1 second** ⚡

### Original Goal vs Achievement
| Metric | Goal | Achieved | Status |
|--------|------|----------|--------|
| **MAE** | 40-50 ETB | **22.87 ETB** | ✅ **75% BETTER!** |
| **R²** | > 0.70 | **0.9894** | ✅ **41% BETTER!** |
| **Training** | < 5 min | **< 1 sec** | ✅ **Much faster!** |

### Interpretation
- The model explains **98.94%** of bill variation
- Typical prediction error: **±23 ETB** (only 1.37% error!)
- **Baseline model** (always predicting mean): ~227 ETB error
- **Our model**: **89.9% improvement** over baseline!

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone or Download Project
```bash
# If using Git
git clone <your-repo-url>
cd energy_bill_ml_project

# Or just download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Plotting and visualizations
- **seaborn** - Statistical visualizations
- **fpdf2** - PDF report generation

---

## 📖 Usage Guide

### Step 1: Generate Training Data
```bash
python data/create_improved_data.py
```

**What this does:**
- Creates 10,000 synthetic but realistic data samples
- Includes realistic relationships between features (physics-based)
- Uses Ethiopian household patterns
- Saves to: `data/improved_energy_bill.csv`

**Expected output:**
```
✅ DATASET CREATED SUCCESSFULLY!
📁 File saved: data/improved_energy_bill.csv
📊 Total samples: 10,000
📈 Features: 9 (house characteristics + appliance usage)
💰 Currency: Ethiopian Birr (ETB)
```

---

### Step 2: Train the Model
```bash
python src/train.py
```

**What this does:**
1. ✅ **Exploratory Data Analysis (EDA)**
   - Statistical summary of all features
   - **Correlation analysis** (AC has strongest correlation: **0.845**!)
   
2. ✅ **Data Preparation**
   - Load 10,000 samples
   - Split: **80% train (8,000)**, **20% test (2,000)**
   
3. ✅ **Feature Scaling**
   - **StandardScaler (Z-score normalization)**
   - Formula: `z = (x - μ) / σ`
   - All features scaled to mean=0, std=1
   - **Critical for gradient descent convergence!**
   
4. ✅ **Model Training**
   - Linear Regression with gradient descent
   - Learns 9 coefficients + 1 intercept
   - Converges in **< 1 second**
   
5. ✅ **Evaluation & Visualization**
   - Calculate R², RMSE, MAE
   - **Generate 7 professional visualizations**
   - Save all results
   
6. ✅ **Save Everything**
   - Model: `models/linear_regression.pkl`
   - Scaler: `models/scaler.pkl`
   - Metadata: `models/metadata.json`
   - **Visualizations**: 7 PNG files in project root

**Expected output:**
```
======================================================================
⚡ ENERGY BILL PREDICTOR - LINEAR REGRESSION
======================================================================

📊 EXPLORATORY DATA ANALYSIS
Dataset Shape: 10000 rows × 10 columns

Statistical Summary:
              house_size_sqm  num_occupants  ...        tv      bill
mean                  125.53           3.51  ...      4.78   1673.74
std                    43.58           1.71  ...      2.18    278.66

📈 Correlation with Bill (Target Variable):
   🔴 ac                    0.845  (VERY STRONG!)
   🟡 season                0.581  (MODERATE)
   🟡 house_size_sqm        0.461  (MODERATE)
   ...

📊 MODEL TRAINING
Training samples: 8,000
Testing samples: 2,000

✅ Features scaled using StandardScaler (Z-score normalization)
✅ Model trained successfully!

🎯 MODEL PERFORMANCE ON TEST SET (2,000 samples):
   • R² Score:  0.9894 (98.94% variance explained)
   • RMSE:      28.68 ETB
   • MAE:       22.87 ETB (EXCEEDED 40-50 ETB GOAL!)
   • Accuracy:  98.5%

📈 IMPROVEMENT vs Basic Model:
   ✓ Error reduced:     90 ETB → 23 ETB (75% improvement!)
   ✓ R² increased:      0.62 → 0.99 (58% improvement!)
   ✓ Features added:    6 → 9 features (+50%)

📊 LEARNED COEFFICIENTS (Feature Importance):
   Feature               Coefficient    Impact
   ───────────────────────────────────────────
   ac                    182.07         HIGHEST (AC dominates!)
   house_size_sqm         88.07         HIGH
   num_occupants          51.44         MEDIUM
   season                 50.30         MEDIUM
   washing_machine        44.03         MEDIUM
   tv                     38.99         LOW
   fans                   26.44         LOW
   lights                 22.08         LOW
   fridge                  0.32         NEGLIGIBLE

💡 KEY INSIGHT: AC usage is BY FAR the dominant cost driver!
   • AC consumes 1,500W vs 10-150W for other appliances (10-150x more!)
   • 1 hour of AC = 5 hours of TV watching (cost-wise)

✅ Model and visualizations saved successfully!
   • models/linear_regression.pkl
   • models/scaler.pkl
   • models/metadata.json
   • 7 visualization PNG files
```

**Time**: ~10-15 seconds (including visualization generation)

---

### Step 3: Make Predictions
```bash
python src/predict.py
```

**What this does:**
- Loads trained model and scaler
- Interactive input for household characteristics
- Predicts monthly bill
- **Shows reliability warnings** for out-of-range inputs
- Provides cost breakdown

**Example interaction:**
```
======================================================================
⚡ ENERGY BILL PREDICTOR
======================================================================

🏠 HOUSE CHARACTERISTICS
House size (square meters): 170
Number of occupants (1-6 people): 4
Season (type 'hot' or 'cool'): cool

⚡ APPLIANCE USAGE (hours per day)
Air Conditioner hours (0-24): 20
Refrigerator hours (0-24): 23
Lights hours (0-24): 18
Fans hours (0-24): 13
Washing Machine hours (0-24): 2
Television hours (0-24): 16

======================================================================
🎯 PREDICTION
======================================================================

💰 PREDICTED MONTHLY BILL: 2,718.09 ETB

⚠️  NOTE: Some inputs are outside training data range:
   • AC usage (20.0 hrs) outside training range (0-20)
   
The model will extrapolate, which may be less accurate.
For best accuracy, use inputs similar to training data.

📊 COST BREAKDOWN (approximate):
   Base Cost:      1,673.92 ETB
   House Size:     +89.83 ETB
   AC Impact:      +624.50 ETB  ← BIGGEST contributor!
   Other factors:  +330.84 ETB

💡 ENERGY SAVING TIP:
   Reducing AC by just 2 hours/day could save ~360 ETB/month!
```

---

## 📁 Project Structure

```
energy_bill_ml_project/
│
├── data/
│   ├── create_improved_data.py    # Data generation script
│   └── improved_energy_bill.csv   # Training data (generated)
│
├── models/
│   ├── linear_regression.pkl      # Trained model (saved after training)
│   ├── scaler.pkl                 # Feature scaler (saved after training)
│   └── metadata.json              # Model metadata (saved after training)
│
├── src/
│   ├── train.py                   # Training script (generates visualizations)
│   └── predict.py                 # Prediction script (interactive)
│
├── generate_complete_report_pdf.py  # PDF report generator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
└── [Visualizations - generated by train.py]
    ├── 1_correlation_heatmap.png
    ├── 2_feature_importance.png
    ├── 3_actual_vs_predicted.png
    ├── 4_residual_plot.png
    ├── 5_distribution_comparison.png
    ├── 6_feature_relationships.png
    └── 7_metrics_summary.png
```

---

## 🎓 Linear Regression Theory

### Model Equation
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₉x₉
```

**Our Learned Equation:**
```
Bill = 1673.92 (base cost)
     + 88.07 × (house_size_sqm)
     + 51.44 × (num_occupants)
     + 50.30 × (season)
     + 182.07 × (ac)              ← HIGHEST!
     + 0.32 × (fridge)             ← LOWEST
     + 22.08 × (lights)
     + 26.44 × (fans)
     + 44.03 × (washing_machine)
     + 38.99 × (tv)
```

Where:
- `ŷ` = predicted bill (ETB)
- `β₀` = intercept (base cost everyone pays)
- `β₁, β₂, ..., β₉` = coefficients (impact of each feature)
- `x₁, x₂, ..., x₉` = features (house size, AC hours, etc.)

### Gradient Descent Algorithm

**Goal**: Find β that minimizes prediction error (cost function)

**Steps**:
1. **Initialize** with random weights β
2. **Calculate** predictions: ŷ = Xβ
3. **Calculate** error (cost): J = (1/2m)Σ(ŷ - y)²
4. **Calculate** gradient: ∂J/∂β (direction of steepest descent)
5. **Update** weights: β = β - α(∂J/∂β)
6. **Repeat** until convergence (cost stops decreasing)

**Parameters**:
- `α` (alpha) = learning rate (step size, typically 0.01)
- `m` = number of training samples (8,000 in our case)
- `∂J/∂β` = gradient (tells us which direction to adjust coefficients)

### Cost Function (Mean Squared Error)
```
J(β) = (1/2m) × Σ(ŷᵢ - yᵢ)²
```

**Why square the errors?**
- Small errors (10 ETB): Squared = 100 (acceptable)
- Medium errors (50 ETB): Squared = 2,500 (penalized 25x)
- Large errors (100 ETB): Squared = 10,000 (penalized 100x)

This heavily penalizes large errors, encouraging consistent predictions.

---

## 📊 Feature Engineering - Why 9 Features?

### Evolution of the Model

**Original 6 features** (Basic Model):
- ac, fridge, lights, fans, washing_machine, tv

**Problem**: 
- Only explains ~62% of variance (R² = 0.62)
- MAE ~90 ETB (too high!)

**Solution: Add 3 NEW features** (Enhanced Model):
1. `house_size_sqm` - Larger houses use more electricity (more appliances)
2. `num_occupants` - More people = longer usage duration
3. `season` - Hot season increases AC/fan usage significantly

**Result**: 
- Explains ~98.94% of variance (R² = 0.9894) ✅
- MAE ~22.87 ETB (75% improvement!) ✅

**This is legitimate feature engineering**, not overfitting, because:
- Features have clear physical meaning
- Test set performance confirms generalization
- Improvements are dramatic and consistent

---

## 🧪 Validation & Testing

### Train-Test Split
- **Training**: 80% (8,000 samples) - Model learns from these
- **Testing**: 20% (2,000 samples) - Model is evaluated on these **ONLY**

### Why This Matters
- Model **never** sees test data during training
- Prevents overfitting (memorizing training data)
- Ensures generalization to new, unseen households
- Test performance (R² = 0.9894) proves the model works!

### Feature Scaling (StandardScaler)
- **Method**: Z-score normalization
- **Formula**: z = (x - μ) / σ
- **Result**: All features have mean=0, std=1
- **Why**: Gradient Descent converges **much faster** with scaled features
- **Example**: AC=20 hours → scaled to 3.43 (3.43 standard deviations above mean)

---

## 📊 Visualizations Generated

The training script (`train.py`) automatically generates 7 professional visualizations:

1. **Correlation Heatmap** (`1_correlation_heatmap.png`)
   - Shows relationships between all features
   - AC has strongest correlation with bill (0.845)

2. **Feature Importance Chart** (`2_feature_importance.png`)
   - Bar chart ranking features by coefficient
   - AC dominates at 182.07, fridge lowest at 0.32

3. **Actual vs Predicted Plot** (`3_actual_vs_predicted.png`)
   - Scatter plot showing R² = 0.9894
   - Points cluster tightly on diagonal line
   - Demonstrates excellent model fit

4. **Residual Plot** (`4_residual_plot.png`)
   - Shows prediction errors distributed randomly around zero
   - No systematic bias (confirms model validity)

5. **Distribution Comparison** (`5_distribution_comparison.png`)
   - Compares actual vs predicted bill distributions
   - Similar shapes confirm model learned patterns correctly

6. **Feature Relationships** (`6_feature_relationships.png`)
   - Six scatter plots showing linear trends
   - AC vs Bill shows strongest upward slope

7. **Performance Metrics Summary** (`7_metrics_summary.png`)
   - Visual dashboard of R², RMSE, MAE values
   - Comparison to baseline model



## 💡 Real-World Applications

1. **Household Budgeting** - Predict bills before they arrive
2. **Energy Efficiency** - Identify high-consumption appliances (AC!)
3. **What-If Analysis** - "If I reduce AC by 2 hours, how much do I save?"
4. **Seasonal Planning** - Prepare for higher bills in hot season
5. **Policy Making** - Understand national consumption patterns
6. **Smart Homes** - Integrate with IoT devices for real-time monitoring

### Energy Saving Recommendations
| Action | Est. Monthly Savings | Difficulty |
|--------|---------------------|------------|
| Reduce AC by 2 hrs/day | ~360 ETB | Easy |
| Switch to LED lights | ~100 ETB | Easy |
| Efficient AC usage (raise temp 1°C) | ~200 ETB | Medium |
| Smart thermostat | ~150 ETB | Medium |

---

=======
## 🎯 Model Limitations & Extrapolation

### Understanding Interpolation vs Extrapolation

**INTERPOLATION** (Reliable):
- Predictions **within** training range (50-200 sqm)
- Model has seen similar data
- **HIGH CONFIDENCE** ✅

**EXTRAPOLATION** (Less Reliable):
- Predictions **outside** training range (e.g., 500 sqm)
- Model extends pattern linearly
- **LOWER CONFIDENCE** ⚠️

### Our Solution: Transparent Warnings
The prediction system displays warnings when inputs are outside training range:
```
⚠️  NOTE: Some inputs are outside training data range:
   • House size (500 sqm) outside training range (50-200)
   • AC usage (24.0 hrs) outside training range (0-20)

The model will extrapolate, which may be less accurate.
```

## 🚨 Common Issues & Solutions

### Issue 1: "Module not found"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue 2: "File not found: improved_energy_bill.csv"
```bash
# Solution: Generate data first
python data/create_improved_data.py
```

### Issue 3: "Model files not found"
```bash
# Solution: Train model first
python src/train.py
```

### Issue 4: "sklearn UserWarning about feature names"
✅ **FIXED!** We now use pandas DataFrames with column names instead of numpy arrays.

### Issue 5: "PDF generation error with bullet character"
✅ **FIXED!** We now use hyphens (-) instead of Unicode bullets (•) for compatibility.

---

## 📚 Learning Outcomes

After completing this project, you have demonstrated understanding of:

✅ **Linear Regression Theory**
- Model equation: ŷ = β₀ + Σ(βᵢxᵢ)
- Coefficient interpretation
- Gradient descent algorithm
- Cost function (MSE)
- Learning rate and convergence

✅ **Machine Learning Pipeline**
- Data generation with realistic patterns
- Exploratory Data Analysis (EDA)
- Train-test split (80/20)
- Feature scaling (StandardScaler)
- Model training and evaluation
- Making predictions on new data

✅ **Feature Engineering**
- Why more features improve accuracy
- How to select relevant features
- Avoiding overfitting through validation
- Physical intuition for feature selection

✅ **Model Evaluation**
- R² Score (98.94% variance explained)
- RMSE and MAE metrics
- Baseline comparison
- Visualization of results

✅ **Real-World Application**
- Ethiopian electricity billing context
- Practical energy-saving recommendations
- Interpretable, explainable results
- Transparent communication of limitations

---

This project is for educational purposes as part of a Supervised Learning course.

---


---

## 🚀 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python data/create_improved_data.py

# 3. Train model (generates visualizations)
python src/train.py

# 4. Make predictions
python src/predict.py

# 5. Generate PDF report (optional)
python generate_complete_report_pdf.py
```

**Total time**: ~2 minutes from start to finish! ⚡
>>>>>>> 95e11ac (Fixed PDF report - resolved image overlap issues)
