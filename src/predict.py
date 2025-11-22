
import os
import sys
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import LinearRegressionModel


def get_user_input():
    """
    Get appliance usage input from user with flexible validation
    
    Returns:
    --------
    input_data : dict
        Dictionary with feature values
    """
    
    print("\n" + "="*70)
    print("📝 ENTER YOUR HOUSEHOLD INFORMATION")
    print("="*70)
    print("💡 Press ENTER to skip any field (defaults to 0)")
    
    input_data = {}
    
    # House characteristics
    print("\n🏠 House Characteristics:")
    print("-"*70)
    
    # House size (flexible: 10-1000 sqm)
    while True:
        try:
            house_input = input("   House size (square meters): ").strip()
            if not house_input:
                house_size = 100  # Default
                print(f"   ℹ️  Using default: {house_size} sqm")
            else:
                house_size = float(house_input)
            
            if house_size < 10:
                print("   ⚠️  House size too small. Minimum: 10 sqm")
                continue
            elif house_size > 1000:
                print(f"   ⚠️  Warning: {house_size} sqm is very large!")
                print("      Training data range: 50-200 sqm")
                print("      Prediction may be less accurate for extreme values.")
                confirm = input("      Continue anyway? (yes/no): ").strip().lower()
                if confirm not in ['yes', 'y']:
                    continue
            elif house_size > 300:
                print(f"   ℹ️  Note: {house_size} sqm is outside typical training range (50-200)")
                print("      Prediction will extrapolate beyond training data.")
            
            input_data['house_size_sqm'] = house_size
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")
    
    # Number of occupants (1-6 as before)
    while True:
        try:
            occupants_input = input("   Number of occupants (1-6 people): ").strip()
            if not occupants_input:
                occupants = 3  # Default
                print(f"   ℹ️  Using default: {occupants} people")
            else:
                occupants = int(occupants_input)
            
            if occupants < 1:
                print("   ⚠️  Must have at least 1 occupant")
                continue
            elif occupants > 6:
                print(f"   ⚠️  {occupants} people is outside training range (1-6)")
                confirm = input("      Continue anyway? (yes/no): ").strip().lower()
                if confirm not in ['yes', 'y']:
                    continue
            
            input_data['num_occupants'] = occupants
            break
        except ValueError:
            print("   ⚠️  Please enter a valid integer")
    
    # Season (keep as is: hot/cool)
    while True:
        season_input = input("   Season (type 'hot' or 'cool'): ").strip().lower()
        if not season_input:
            season_input = 'cool'
            print(f"   ℹ️  Using default: {season_input}")
        
        if season_input in ['hot', 'cool']:
            input_data['season'] = 1 if season_input == 'hot' else 0
            break
        else:
            print("   ⚠️  Please type 'hot' or 'cool'")
    
    # Appliance usage (ALL flexible: 0-24 hours)
    print("\n⚡ Appliance Daily Usage (hours per day):")
    print("-"*70)
    print("💡 Enter any value from 0 to 24 hours")
    print("   Press ENTER to skip (defaults to 0)\n")
    
    appliances = {
        'ac': 'Air Conditioner',
        'fridge': 'Refrigerator',
        'lights': 'Lights',
        'fans': 'Fans',
        'washing_machine': 'Washing Machine',
        'tv': 'Television'
    }
    
    for key, name in appliances.items():
        while True:
            try:
                value_input = input(f"   {name} (0-24 hours): ").strip()
                
                if not value_input:
                    value = 0
                    print(f"   ℹ️  Using default: 0 hours")
                else:
                    value = float(value_input)
                
                # Validation
                if value < 0:
                    print("   ⚠️  Hours cannot be negative")
                    continue
                elif value > 24:
                    print(f"   ⚠️  Warning: {value} hours exceeds 24 hours/day!")
                    confirm = input("      Continue anyway? (yes/no): ").strip().lower()
                    if confirm not in ['yes', 'y']:
                        continue
                
                # Special note for values outside training range
                if key == 'fridge' and value < 20:
                    print(f"   ℹ️  Note: Fridges typically run 23-24 hours/day in training data")
                    print(f"      Your input ({value} hours) is unusual but allowed.")
                elif key == 'ac' and value > 20:
                    print(f"   ℹ️  Note: {value} hours is higher than typical training range (0-20)")
                    print(f"      Prediction will extrapolate.")
                
                input_data[key] = value
                break
            except ValueError:
                print("   ⚠️  Please enter a valid number")
    
    return input_data


def predict_bill():
    """Main prediction function"""
    
    print("\n" + "="*70)
    print("⚡ ENERGY BILL PREDICTOR")
    print("="*70)
    print("Author: Yeabsira Samuel")
    print("Course: Supervised Learning - Linear Regression")
    print("Currency: Ethiopian Birr (ETB)")
    print("="*70)
    
    # Load model
    print("\n📂 Loading trained model...")
    
    try:
        model = LinearRegressionModel.load('models/linear_regression_model.pkl')
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print("✅ Feature names loaded")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Model files not found!")
        print("   Please run: python src/train.py")
        return
    
    # Get user input
    input_data = get_user_input()
    
    # Prepare data for prediction
    print("\n" + "="*70)
    print("🔄 PROCESSING INPUT")
    print("="*70)
    
    # Create input array in correct order
    X_input = np.array([[input_data[feature] for feature in feature_names]])
    
    print("\n📊 Your Input Summary:")
    print("-"*70)
    for feature in feature_names:
        value = input_data[feature]
        if feature == 'season':
            display = "Hot ☀️" if value == 1 else "Cool ❄️"
            print(f"   {feature:<20}: {display}")
        elif feature == 'house_size_sqm':
            print(f"   {feature:<20}: {value:.0f} sqm")
        elif feature == 'num_occupants':
            print(f"   {feature:<20}: {value:.0f} people")
        else:
            print(f"   {feature:<20}: {value:.1f} hours/day")
    
    # Scale the input
    X_scaled = scaler.transform(X_input)
    print("\n✅ Input scaled successfully")
    
    # Make prediction
    print("\n" + "="*70)
    print("🎯 PREDICTION")
    print("="*70)
    
    predicted_bill = model.predict(X_scaled)[0]
    
    print(f"\n💰 PREDICTED MONTHLY BILL: {predicted_bill:.2f} ETB")
    
    # Check if prediction is reliable
    print("\n📊 PREDICTION RELIABILITY:")
    print("-"*70)
    
    outside_range = False
    warnings = []
    
    # Check each feature
    if input_data['house_size_sqm'] < 50 or input_data['house_size_sqm'] > 200:
        warnings.append(f"House size ({input_data['house_size_sqm']:.0f} sqm) outside training range (50-200)")
        outside_range = True
    
    if input_data['ac'] > 20:
        warnings.append(f"AC usage ({input_data['ac']:.1f} hrs) outside training range (0-20)")
        outside_range = True
    
    if input_data['fridge'] < 23:
        warnings.append(f"Fridge usage ({input_data['fridge']:.1f} hrs) outside typical range (23-24)")
        outside_range = True
    
    if warnings:
        print("   ⚠️  NOTE: Some inputs are outside training data range:")
        for warning in warnings:
            print(f"      • {warning}")
        print("\n   The model will extrapolate, which may be less accurate.")
        print("   For best accuracy, use inputs similar to training data.")
    else:
        print("   ✅ All inputs within training range - prediction highly reliable!")
    
    # Show breakdown by feature
    print("\n" + "="*70)
    print("📊 COST BREAKDOWN (Approximate)")
    print("="*70)
    
    # Get coefficients
    coefficients = model.model.coef_
    intercept = model.model.intercept_
    
    print(f"\n   Base Cost (Intercept): {intercept:.2f} ETB")
    print("\n   Feature Contributions:")
    print("-"*70)
    
    total_contribution = 0
    for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
        # Approximate contribution (scaled)
        contribution = coef * X_scaled[0][i]
        total_contribution += contribution
        
        emoji = "🔴" if abs(contribution) > 100 else "🟡" if abs(contribution) > 50 else "🟢"
        sign = "+" if contribution >= 0 else ""
        print(f"   {emoji} {feature:<20}: {sign}{contribution:.2f} ETB")
    
    print("-"*70)
    print(f"   Total: {predicted_bill:.2f} ETB")
    
    # Give recommendations
    print("\n" + "="*70)
    print("💡 ENERGY SAVING TIPS")
    print("="*70)
    
    tips = []
    
    if input_data.get('ac', 0) > 10:
        savings = (input_data['ac'] - 8) * 182.07  # Approximate per coefficient
        tips.append(f"• Reduce AC to 8 hours/day → Save ~{savings:.0f} ETB/month")
    
    if input_data.get('lights', 0) > 8:
        tips.append("• Use LED bulbs and turn off unused lights → Save ~50 ETB/month")
    
    if input_data.get('washing_machine', 0) > 3:
        tips.append("• Wash full loads only → Save ~40 ETB/month")
    
    if input_data.get('fans', 0) > 12:
        tips.append("• Use fans instead of AC when possible → Save ~150 ETB/month")
    
    if input_data.get('tv', 0) > 8:
        tips.append("• Reduce TV usage by 2 hours → Save ~30 ETB/month")
    
    if input_data.get('season') == 1:
        tips.append("• Close curtains during hot hours → Reduce AC usage")
        tips.append("• Use ceiling fans to circulate air → Lower AC need")
    
    if tips:
        for tip in tips:
            print(f"\n   {tip}")
    else:
        print("\n   ✅ Your usage is already efficient! Keep it up!")
    
    # Ask if user wants to try again
    print("\n" + "="*70)
    retry = input("\n🔄 Predict another bill? (yes/no): ").strip().lower()
    
    if retry in ['yes', 'y']:
        predict_bill()  # Recursive call
    else:
        print("\n" + "="*70)
        print("👋 Thank you for using Energy Bill Predictor!")
        print("="*70 + "\n")


if __name__ == "__main__":
    predict_bill()