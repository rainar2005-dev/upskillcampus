import pandas as pd
import numpy as np

def generate_sample_csv():
    # Define sample data for visualization and prediction testing
    crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Jute', 'Tea', 'Coffee']
    states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'West Bengal', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh']
    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']
    
    data = []
    for _ in range(500):
        crop = np.random.choice(crops)
        state = np.random.choice(states)
        season = np.random.choice(seasons)
        area = np.random.randint(100, 5000)
        cost = np.random.randint(1000, 50000)
        # Random production logic: production depends on area, cost, and random factor
        production = (area * np.random.uniform(0.5, 2.5)) + (cost / 100)
        
        data.append([crop, state, season, area, cost, round(production, 2)])
    
    df = pd.DataFrame(data, columns=['Crop', 'State', 'Season', 'Area', 'Cost', 'Production'])
    
    # Save as CSV
    df.to_csv('crop_production_data.csv', index=False)
    print("Sample CSV file 'crop_production_data.csv' has been generated.")

if __name__ == "__main__":
    generate_sample_csv()
