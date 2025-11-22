"""
Linear Regression Model with Gradient Descent
==============================================
Course: Supervised Learning - Linear Regression

Author: Yeabsira Samuel
Currency: Ethiopian Birr (ETB)

THEORY:
-------
Linear Regression finds the best-fit line through data points.

Model Equation:
    ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    
    Where:
    ŷ = predicted bill (ETB)
    β₀ = intercept (base cost)
    β₁, β₂, ..., βₙ = coefficients (weights)
    x₁, x₂, ..., xₙ = features (hours of usage)

Gradient Descent Algorithm:
    1. Start with random weights (β)
    2. Calculate predictions: ŷ = Xβ
    3. Calculate error (cost): J = (1/2m) Σ(ŷ - y)²
    4. Update weights: β = β - α(∂J/∂β)
    5. Repeat until convergence

Cost Function (Mean Squared Error):
    J(β) = (1/2m) Σ(ŷᵢ - yᵢ)²
    
    Where:
    m = number of samples
    ŷᵢ = predicted value
    yᵢ = actual value
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


class LinearRegressionModel:
    """
    Linear Regression Model with Gradient Descent
    
    This class implements Linear Regression for predicting electricity bills
    from appliance usage patterns.
    """
    
    def __init__(self):
        """Initialize the Linear Regression model"""
        self.model = LinearRegression()
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the Linear Regression model
        
        Parameters:
        -----------
        X_train : numpy array
            Training features (scaled)
        y_train : numpy array
            Training targets (bills in ETB)
        feature_names : list
            Names of features
        """
        
        print("\n" + "="*70)
        print("🎓 LINEAR REGRESSION THEORY")
        print("="*70)
        
        print("\n📐 MODEL EQUATION:")
        print("   ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
        print()
        print("   Where:")
        print("   • ŷ = predicted bill (ETB)")
        print("   • β₀ = intercept (base cost)")
        print("   • β₁, β₂, ..., βₙ = coefficients (impact of each feature)")
        print("   • x₁, x₂, ..., xₙ = features (appliance usage hours)")
        
        print("\n🎯 GRADIENT DESCENT ALGORITHM:")
        print("="*70)
        print("Goal: Find β that minimizes prediction error")
        print()
        print("Steps:")
        print("   1️⃣  Initialize: Start with random weights β")
        print("   2️⃣  Predict: Calculate ŷ = Xβ")
        print("   3️⃣  Error: Calculate cost J = (1/2m)Σ(ŷ - y)²")
        print("   4️⃣  Update: β = β - α(∂J/∂β)")
        print("   5️⃣  Repeat: Until cost stops decreasing")
        print()
        print("Where:")
        print("   • α (alpha) = learning rate (step size)")
        print("   • ∂J/∂β = gradient (direction of steepest descent)")
        print("   • m = number of training samples")
        
        print("\n📊 COST FUNCTION (Mean Squared Error):")
        print("="*70)
        print("   J(β) = (1/2m) Σ(ŷᵢ - yᵢ)²")
        print()
        print("   This measures how far predictions are from actual values.")
        print("   Gradient Descent minimizes this cost function.")
        
        print("\n🔄 TRAINING IN PROGRESS...")
        print("="*70)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names
        self.is_trained = True
        
        # Get learned parameters
        intercept = self.model.intercept_
        coefficients = self.model.coef_
        
        print("✅ Training completed!")
        
        print("\n📊 LEARNED PARAMETERS:")
        print("="*70)
        print(f"   β₀ (Intercept): {intercept:.2f} ETB")
        print()
        print("   Coefficients (β₁, β₂, ..., βₙ):")
        if feature_names:
            for i, (name, coef) in enumerate(zip(feature_names, coefficients), 1):
                emoji = "🔴" if abs(coef) > 100 else "🟡" if abs(coef) > 50 else "🟢"
                print(f"      {emoji} β{i} ({name:<20}): {coef:>8.2f}")
        else:
            for i, coef in enumerate(coefficients, 1):
                print(f"      β{i}: {coef:.2f}")
        
        print("\n💡 Interpretation:")
        print("   • Positive coefficient = feature increases bill")
        print("   • Negative coefficient = feature decreases bill")
        print("   • Larger |coefficient| = stronger impact on bill")
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy array
            Features to predict on
            
        Returns:
        --------
        predictions : numpy array
            Predicted bills (ETB)
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : numpy array
            Testing features
        y_test : numpy array
            Actual bills (ETB)
            
        Returns:
        --------
        metrics : dict
            R², RMSE, MAE
        """
        
        print("\n" + "="*70)
        print("📊 MODEL EVALUATION")
        print("="*70)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\n📈 Performance Metrics:")
        print("-"*70)
        
        # R² Score
        print(f"\n   R² Score: {r2:.4f}")
        print("   ├─ Interpretation: Percentage of variance explained")
        if r2 > 0.9:
            print("   └─ ⭐⭐⭐ Excellent fit!")
        elif r2 > 0.7:
            print("   └─ ⭐⭐ Good fit!")
        elif r2 > 0.5:
            print("   └─ ⭐ Acceptable fit")
        else:
            print("   └─ ⚠️  Poor fit - consider more features")
        
        # RMSE
        print(f"\n   RMSE: {rmse:.2f} ETB")
        print("   ├─ Interpretation: Average prediction error (penalizes large errors)")
        print(f"   └─ On average, predictions are off by ±{rmse:.2f} ETB")
        
        # MAE
        print(f"\n   MAE: {mae:.2f} ETB")
        print("   ├─ Interpretation: Average absolute error")
        print(f"   └─ Typical error: {mae:.2f} ETB")
        
        # Compare with baseline
        baseline_error = np.mean(np.abs(y_test - np.mean(y_test)))
        improvement = ((baseline_error - mae) / baseline_error) * 100
        
        print(f"\n   Baseline (predicting mean): {baseline_error:.2f} ETB")
        print(f"   Improvement: {improvement:.1f}%")
        
        print("\n" + "="*70)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
    
    def save(self, filepath='models/linear_regression_model.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"\n✅ Model saved: {filepath}")
    
    @staticmethod
    def load(filepath='models/linear_regression_model.pkl'):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded: {filepath}")
        return model
    
    def get_feature_importance(self):
        """
        Get feature importance based on absolute coefficient values
        
        Returns:
        --------
        importance : dict
            Feature names and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        if self.feature_names is None:
            return None
        
        # Absolute value of coefficients
        importance = {
            name: abs(coef) 
            for name, coef in zip(self.feature_names, self.model.coef_)
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance


if __name__ == "__main__":
    # Test the model
    print("Linear Regression Model Module")
    print("Run train.py to train the model")