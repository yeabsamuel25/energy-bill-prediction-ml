"""
VISUALIZATION SCRIPT - FIXED VERSION
=====================================
Creates professional visualizations for Linear Regression project:
1. Correlation Heatmap
2. Feature Importance Bar Chart
3. Actual vs Predicted (Linear Regression Line)
4. Residual Plot
5. Distribution Plots
6. Feature Relationships with Bill

Author: Yeabsira Samuel
Course: Supervised Learning - Linear Regression
Currency: Ethiopian Birr (ETB)
"""

import sys
import os

# Add project root to Python path (MUST be before other imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_all_visualizations():
    """Generate all visualization plots"""
    
    print("=" * 70)
    print("📊 CREATING VISUALIZATIONS FOR LINEAR REGRESSION PROJECT")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('data/improved_energy_bill.csv')
    print(f"✅ Loaded: {len(df):,} samples")
    
    # Load trained model
    print("📂 Loading trained model...")
    with open('models/linear_regression_model.pkl', 'rb') as f:
        model_wrapper = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("✅ Model loaded")
    
    # Get the actual sklearn model from the wrapper
    sklearn_model = model_wrapper.model
    
    # Prepare data - use correct column name
    target_col = 'monthly_bill_etb' if 'monthly_bill_etb' in df.columns else 'bill'
    X = df[feature_names]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions using the sklearn model
    y_pred = sklearn_model.predict(X_test_scaled)
    
    print("\n🎨 Creating visualizations...")
    print("-" * 70)
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # =================================================================
    # VISUALIZATION 1: CORRELATION HEATMAP
    # =================================================================
    print("\n1️⃣  Creating correlation heatmap...")
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        mask=mask,
        vmin=-1,
        vmax=1
    )
    
    plt.title('📊 Correlation Heatmap - Energy Bill Features\n', 
              fontsize=16, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('visualizations/1_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/1_correlation_heatmap.png")
    
    # =================================================================
    # VISUALIZATION 2: FEATURE IMPORTANCE BAR CHART
    # =================================================================
    print("\n2️⃣  Creating feature importance chart...")
    
    # Get coefficients from the sklearn model (not the wrapper)
    coefficients = np.abs(sklearn_model.coef_)
    
    # Sort by importance
    sorted_idx = np.argsort(coefficients)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_coefficients = coefficients[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_features)))
    
    bars = plt.barh(sorted_features, sorted_coefficients, color=colors, edgecolor='black')
    
    # Add value labels
    for i, (bar, coef) in enumerate(zip(bars, sorted_coefficients)):
        plt.text(coef + 5, i, f'{coef:.2f}', 
                va='center', fontweight='bold', fontsize=10)
    
    plt.xlabel('Absolute Coefficient Value (Impact on Bill)', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('📊 Feature Importance - Linear Regression\n(Higher = Stronger Impact on Bill)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/2_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/2_feature_importance.png")
    
    # =================================================================
    # VISUALIZATION 3: ACTUAL VS PREDICTED (LINEAR REGRESSION LINE)
    # =================================================================
    print("\n3️⃣  Creating actual vs predicted plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction (y = x)')
    
    # Add R² score
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('Actual Bill (ETB)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Bill (ETB)', fontsize=12, fontweight='bold')
    plt.title('📊 Actual vs Predicted Bills - Linear Regression\n', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/3_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/3_actual_vs_predicted.png")
    
    # =================================================================
    # VISUALIZATION 4: RESIDUAL PLOT
    # =================================================================
    print("\n4️⃣  Creating residual plot...")
    
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    plt.xlabel('Predicted Bill (ETB)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted) ETB', fontsize=12, fontweight='bold')
    plt.title('📊 Residual Plot - Model Error Analysis\n', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/4_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/4_residual_plot.png")
    
    # =================================================================
    # VISUALIZATION 5: DISTRIBUTION OF PREDICTIONS
    # =================================================================
    print("\n5️⃣  Creating prediction distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(y_test, bins=30, alpha=0.6, color='blue', label='Actual', edgecolor='black')
    axes[0].hist(y_pred, bins=30, alpha=0.6, color='red', label='Predicted', edgecolor='black')
    axes[0].set_xlabel('Bill (ETB)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Actual vs Predicted Bills', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (ETB)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Residuals (Errors)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/5_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/5_distributions.png")
    
    # =================================================================
    # VISUALIZATION 6: TOP FEATURES VS BILL (SCATTER PLOTS)
    # =================================================================
    print("\n6️⃣  Creating feature relationships plot...")
    
    # Get top 6 features by importance
    top_6_idx = sorted_idx[:6]
    top_features = [feature_names[i] for i in top_6_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        axes[i].scatter(df[feature], df[target_col], alpha=0.3, s=20, edgecolor='black', linewidth=0.3)
        axes[i].set_xlabel(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Bill (ETB)', fontsize=10, fontweight='bold')
        axes[i].set_title(f'{feature.replace("_", " ").title()} vs Bill', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation value
        corr = df[[feature, target_col]].corr().iloc[0, 1]
        axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/6_feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/6_feature_relationships.png")
    
    # =================================================================
    # VISUALIZATION 7: ERROR METRICS SUMMARY
    # =================================================================
    print("\n7️⃣  Creating error metrics summary...")
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['R² Score', 'RMSE (ETB)', 'MAE (ETB)']
    values = [r2, rmse, mae]
    colors_list = ['green', 'orange', 'blue']
    
    bars = ax.bar(metrics, values, color=colors_list, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('📊 Model Performance Metrics\n', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation text
    interpretation = f"""
    R² = {r2:.4f}: Model explains {r2*100:.2f}% of variance ✅
    RMSE = {rmse:.2f} ETB: Average prediction error ✅
    MAE = {mae:.2f} ETB: Typical error in predictions ✅
    """
    ax.text(0.5, -0.2, interpretation, 
            transform=ax.transAxes, 
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/7_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: visualizations/7_performance_metrics.png")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📁 Saved in: visualizations/ folder")
    print("\n📊 Created Visualizations:")
    print("   1️⃣  1_correlation_heatmap.png - Shows relationships between all features")
    print("   2️⃣  2_feature_importance.png - Shows which features impact bill most")
    print("   3️⃣  3_actual_vs_predicted.png - Linear regression line (y=x)")
    print("   4️⃣  4_residual_plot.png - Shows prediction errors")
    print("   5️⃣  5_distributions.png - Compares actual vs predicted distributions")
    print("   6️⃣  6_feature_relationships.png - Top 6 features vs bill")
    print("   7️⃣  7_performance_metrics.png - R², RMSE, MAE summary")
    
    print("\n💡 How to use these visualizations:")
    print("   • Open the images in your presentation")
    print("   • Explain each visualization to your teacher")
    print(f"   • Show how R² = {r2:.4f} means excellent fit")
    print("   • Point out which feature has highest correlation")
    print("   • Show the actual vs predicted plot proves good predictions")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    create_all_visualizations()