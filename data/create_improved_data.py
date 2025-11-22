"""
IMPROVED DATA GENERATOR
=======================
Creates realistic electricity bill data with 9 features (instead of 6)
This improves prediction accuracy from MAE ~90 ETB to ~45 ETB

Author: Yeabsira Samuel
Course: Supervised Learning - Linear Regression
Currency: Ethiopian Birr (ETB)

NEW FEATURES ADDED:
1. house_size_sqm - House size affects total consumption
2. num_occupants - More people = more usage
3. season - Summer (1) = more AC usage, Winter (0) = less AC
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def create_improved_dataset(n_samples=10000):
    """
    Create improved electricity bill dataset
    
    Features:
    ---------
    Original 6:
    - ac, fridge, lights, fans, washing_machine, tv (hours/day)
    
    NEW 3:
    - house_size_sqm: 50-200 square meters
    - num_occupants: 1-6 people
    - season: 0 (cool) or 1 (hot)
    
    Target:
    -------
    - Bill: Monthly electricity bill in Ethiopian Birr (ETB)
    
    Formula:
    --------
    Bill = base_cost 
           + (50 * ac) 
           + (0.3 * fridge) 
           + (10 * lights) 
           + (8 * fans) 
           + (40 * washing_machine) 
           + (18 * tv)
           + (2 * house_size_sqm)      # NEW: Bigger house = more appliances
           + (30 * num_occupants)       # NEW: More people = more usage
           + (100 * season)             # NEW: Hot season = more AC/fans
           + random_noise
    """
    
    print("="*70)
    print("🏗️  CREATING IMPROVED ELECTRICITY BILL DATASET")
    print("="*70)
    
    data = {}
    
    # Generate house characteristics (NEW!)
    print("\n📊 Generating house characteristics...")
    data['house_size_sqm'] = np.random.randint(50, 201, n_samples)  # 50-200 sqm
    data['num_occupants'] = np.random.randint(1, 7, n_samples)      # 1-6 people
    data['season'] = np.random.choice([0, 1], n_samples)            # 0=cool, 1=hot
    
    print(f"   ✓ house_size_sqm: {data['house_size_sqm'].min()}-{data['house_size_sqm'].max()} sqm")
    print(f"   ✓ num_occupants: {data['num_occupants'].min()}-{data['num_occupants'].max()} people")
    print(f"   ✓ season: 0 (cool) or 1 (hot)")
    
    # Generate appliance usage (influenced by house size and occupants)
    print("\n🏠 Generating appliance usage (hours per day)...")
    
    # AC: More usage in hot season and bigger houses
    data['ac'] = np.clip(
        np.random.normal(6, 3, n_samples) + 
        data['season'] * 4 +                    # +4 hours in hot season
        (data['house_size_sqm'] - 125) / 50,   # Bigger house = more AC
        0, 24
    ).astype(int)
    
    # Fridge: Always 24 hours (slight variation for realism)
    data['fridge'] = np.clip(
        np.random.normal(24, 0.5, n_samples),
        23, 24
    ).astype(int)
    
    # Lights: More usage with more occupants
    data['lights'] = np.clip(
        np.random.normal(6, 2, n_samples) + 
        (data['num_occupants'] - 3) * 0.5,     # More people = more lights
        0, 24
    ).astype(int)
    
    # Fans: More in hot season
    data['fans'] = np.clip(
        np.random.normal(8, 3, n_samples) + 
        data['season'] * 3,                    # +3 hours in hot season
        0, 24
    ).astype(int)
    
    # Washing Machine: More with more occupants
    data['washing_machine'] = np.clip(
        np.random.normal(2, 1, n_samples) + 
        (data['num_occupants'] - 3) * 0.3,     # More people = more laundry
        0, 8
    ).astype(int)
    
    # TV: More with more occupants
    data['tv'] = np.clip(
        np.random.normal(5, 2, n_samples) + 
        (data['num_occupants'] - 3) * 0.5,     # More people = more TV time
        0, 16
    ).astype(int)
    
    print(f"   ✓ AC: {data['ac'].min()}-{data['ac'].max()} hours/day")
    print(f"   ✓ Fridge: {data['fridge'].min()}-{data['fridge'].max()} hours/day")
    print(f"   ✓ Lights: {data['lights'].min()}-{data['lights'].max()} hours/day")
    print(f"   ✓ Fans: {data['fans'].min()}-{data['fans'].max()} hours/day")
    print(f"   ✓ Washing Machine: {data['washing_machine'].min()}-{data['washing_machine'].max()} hours/day")
    print(f"   ✓ TV: {data['tv'].min()}-{data['tv'].max()} hours/day")
    
    # Calculate electricity bill (Ethiopian Birr)
    print("\n💰 Calculating electricity bills...")
    
    # Realistic Ethiopian electricity pricing (ETB/hour)
    base_cost = 600           # Fixed monthly charges
    ac_rate = 50              # Most expensive
    fridge_rate = 0.3         # Always on but efficient
    lights_rate = 10
    fans_rate = 8
    washing_machine_rate = 40
    tv_rate = 18
    house_size_rate = 2       # Cost per sqm
    occupant_rate = 30        # Cost per person
    season_rate = 100         # Extra cost in hot season
    
    data['bill'] = (
        base_cost +
        ac_rate * data['ac'] +
        fridge_rate * data['fridge'] +
        lights_rate * data['lights'] +
        fans_rate * data['fans'] +
        washing_machine_rate * data['washing_machine'] +
        tv_rate * data['tv'] +
        house_size_rate * data['house_size_sqm'] +
        occupant_rate * data['num_occupants'] +
        season_rate * data['season'] +
        np.random.normal(0, 30, n_samples)  # Small random variation
    ).astype(int)
    
    # Ensure no negative bills
    data['bill'] = np.clip(data['bill'], 800, 3000)
    
    print(f"   ✓ Bills range: {data['bill'].min()}-{data['bill'].max()} ETB")
    print(f"   ✓ Average bill: {data['bill'].mean():.2f} ETB")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns (features first, then target)
    columns_order = [
        'house_size_sqm', 'num_occupants', 'season',
        'ac', 'fridge', 'lights', 'fans', 'washing_machine', 'tv',
        'bill'
    ]
    df = df[columns_order]
    
    return df


def main():
    """Generate and save improved dataset"""
    
    print("\n" + "="*70)
    print("📝 SUPERVISED LEARNING PROJECT - IMPROVED DATA GENERATION")
    print("="*70)
    
    # Create dataset
    df = create_improved_dataset(n_samples=10000)
    
    # Save to CSV
    output_file = 'data/improved_energy_bill.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✅ DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 File saved: {output_file}")
    print(f"📊 Total samples: {len(df):,}")
    print(f"📊 Total features: {len(df.columns) - 1}")
    print(f"🎯 Target variable: bill (ETB)")
    
    print("\n📊 Feature Summary:")
    print("-" * 70)
    print(f"{'Feature':<25} {'Min':<10} {'Max':<10} {'Mean':<10}")
    print("-" * 70)
    for col in df.columns:
        print(f"{col:<25} {df[col].min():<10} {df[col].max():<10} {df[col].mean():<10.2f}")
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS:")
    print("="*70)
    print("1. Run: python src/train.py")
    print("2. Run: python src/predict.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()