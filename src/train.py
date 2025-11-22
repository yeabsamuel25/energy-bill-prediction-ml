"""
TRAINING SCRIPT
===============
Trains Linear Regression model on electricity bill data

Author: Yeabsira Samuel
Course: Supervised Learning - Linear Regression
Currency: Ethiopian Birr (ETB)

This script:
1. Loads and explores data
2. Trains Linear Regression model
3. Evaluates performance
4. Saves trained model
"""

import os
import sys
import pickle

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data, explore_data
from src.model import LinearRegressionModel


def train_model():
    """Complete training pipeline"""
    
    print("\n" + "="*70)
    print("⚡ ENERGY BILL PREDICTOR - LINEAR REGRESSION")
    print("="*70)
    print("Author: Yeabsira Samuel")
    print("Course: Supervised Learning")
    print("Algorithm: Linear Regression with Gradient Descent")
    print("Currency: Ethiopian Birr (ETB)")
    print("="*70)
    
    # STEP 1: Explore Data
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    explore_data('data/improved_energy_bill.csv')
    
    input("\n⏸️  Press ENTER to continue to data loading...")
    
    # STEP 2: Load Data
    print("\n" + "="*70)
    print("STEP 2: LOAD & PREPARE DATA")
    print("="*70)
    
    X_train, X_test, y_train, y_test, scaler, feature_names = load_data(
        filepath='data/improved_energy_bill.csv',
        test_size=0.2,
        random_state=42
    )
    
    if X_train is None:
        print("\n❌ Data loading failed. Please run: python data/create_improved_data.py")
        return
    
    input("\n⏸️  Press ENTER to continue to training...")
    
    # STEP 3: Train Model
    print("\n" + "="*70)
    print("STEP 3: TRAIN LINEAR REGRESSION MODEL")
    print("="*70)
    
    model = LinearRegressionModel()
    model.train(X_train, y_train, feature_names)
    
    input("\n⏸️  Press ENTER to continue to evaluation...")
    
    # STEP 4: Evaluate Model
    print("\n" + "="*70)
    print("STEP 4: EVALUATE MODEL PERFORMANCE")
    print("="*70)
    
    metrics = model.evaluate(X_test, y_test)
    
    # STEP 5: Feature Importance
    print("\n" + "="*70)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    importance = model.get_feature_importance()
    
    if importance:
        print("\n📊 Features ranked by importance:")
        print("-"*70)
        max_importance = max(importance.values())
        
        for i, (feature, score) in enumerate(importance.items(), 1):
            bar_length = int((score / max_importance) * 40)
            bar = "█" * bar_length
            print(f"{i}. {feature:<20} {bar} {score:.2f}")
        
        print("\n💡 Interpretation:")
        print("   Features with higher bars have more impact on the bill.")
    
    input("\n⏸️  Press ENTER to continue to saving model...")
    
    # STEP 6: Save Model
    print("\n" + "="*70)
    print("STEP 6: SAVE TRAINED MODEL")
    print("="*70)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/linear_regression_model.pkl')
    
    # Save scaler (needed for predictions)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved: models/scaler.pkl")
    
    # Save feature names (needed for predictions)
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✅ Feature names saved: models/feature_names.pkl")
    
    # STEP 7: Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Final Model Performance:")
    print(f"   • R² Score: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}% variance explained)")
    print(f"   • RMSE: {metrics['rmse']:.2f} ETB")
    print(f"   • MAE: {metrics['mae']:.2f} ETB")
    
    print("\n📁 Saved Files:")
    print("   • models/linear_regression_model.pkl")
    print("   • models/scaler.pkl")
    print("   • models/feature_names.pkl")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: python src/predict.py")
    print("   2. Input your appliance usage")
    print("   3. Get predicted bill in ETB")
    
    print("\n" + "="*70)
    print("🎉 READY TO MAKE PREDICTIONS!")
    print("="*70 + "\n")


if __name__ == "__main__":
    train_model()