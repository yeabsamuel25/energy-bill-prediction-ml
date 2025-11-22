"""
Data Loader Module
==================
Loads and explores electricity bill data for Linear Regression

Author: Yeabsira Samuel
Course: Supervised Learning - Linear Regression
Currency: Ethiopian Birr (ETB)

Functions:
----------
- load_data(): Load dataset and split into train/test
- explore_data(): Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='data/improved_energy_bill.csv', test_size=0.2, random_state=42):
    """
    Load electricity bill dataset and prepare for training
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    test_size : float
        Proportion of data for testing (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test : Feature matrices (scaled)
    y_train, y_test : Target vectors (bills in ETB)
    scaler : StandardScaler object (for inverse transform)
    feature_names : List of feature names
    """
    
    print("\n" + "="*70)
    print("📂 LOADING DATA")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded: {filepath}")
        print(f"   📊 Total samples: {len(df):,}")
        print(f"   📊 Total features: {len(df.columns) - 1}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {filepath}")
        print("   💡 Run: python data/create_improved_data.py")
        return None, None, None, None, None, None
    
    # Separate features (X) and target (y)
    X = df.drop('bill', axis=1)
    y = df['bill']
    
    feature_names = X.columns.tolist()
    
    print(f"\n📊 Features (X): {feature_names}")
    print(f"🎯 Target (y): bill (ETB)")
    
    # Split into training and testing sets
    print(f"\n🔀 Splitting data...")
    print(f"   📊 Training: {(1-test_size)*100:.0f}%")
    print(f"   📊 Testing: {test_size*100:.0f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"   ✅ Training samples: {len(X_train):,}")
    print(f"   ✅ Testing samples: {len(X_test):,}")
    
    # Feature Scaling (IMPORTANT for Gradient Descent!)
    print(f"\n⚖️  FEATURE SCALING")
    print("="*70)
    print("Why we need feature scaling:")
    print("  • Features have different ranges (e.g., fridge: 23-24, ac: 0-18)")
    print("  • Gradient Descent converges faster with scaled features")
    print("  • Prevents features with large values from dominating")
    print()
    print("Method: Standardization (Z-score normalization)")
    print("  Formula: z = (x - μ) / σ")
    print("  Where: μ = mean, σ = standard deviation")
    print("  Result: Features have mean=0, std=1")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Features scaled successfully!")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_names


def explore_data(filepath='data/improved_energy_bill.csv'):
    """
    Perform Exploratory Data Analysis (EDA)
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    """
    
    print("\n" + "="*70)
    print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {filepath}")
        return
    
    # Basic info
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Statistical summary
    print("\n📈 Statistical Summary:")
    print("-"*70)
    print(df.describe().round(2).to_string())
    
    # Check for missing values
    print("\n🔎 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values!")
    else:
        print(missing[missing > 0])
    
    # Correlation with target (bill)
    print("\n📊 Correlation with Bill (ETB):")
    print("-"*70)
    correlations = df.corr()['bill'].drop('bill').sort_values(ascending=False)
    for feature, corr in correlations.items():
        emoji = "🔴" if abs(corr) > 0.7 else "🟡" if abs(corr) > 0.4 else "🟢"
        print(f"   {emoji} {feature:<20} {corr:>6.3f}")
    
    print("\n💡 Interpretation:")
    print("   🔴 Strong correlation (|r| > 0.7)")
    print("   🟡 Moderate correlation (|r| > 0.4)")
    print("   🟢 Weak correlation (|r| ≤ 0.4)")
    
    # Sample data
    print("\n📋 Sample Data (first 5 rows):")
    print("-"*70)
    print(df.head().to_string(index=False))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Test the module
    explore_data()
    print("\n")
    X_train, X_test, y_train, y_test, scaler, features = load_data()
    
    if X_train is not None:
        print("\n✅ Data loading successful!")
        print(f"   📊 Training features shape: {X_train.shape}")
        print(f"   📊 Testing features shape: {X_test.shape}")