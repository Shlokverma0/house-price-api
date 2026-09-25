import joblib
import pandas as pd

m = joblib.load('models/house_model.pkl')
c = joblib.load('models/house_columns.pkl')

base = {'BHK': 3, 'Size_in_SqFt': 1500, 'Price_per_SqFt': 5000,
        'Year_Built': 2015, 'Parking_Space': 1}

for loc in ['New Delhi', 'Mumbai', 'Bangalore', 'Haridwar', 'Nagpur']:
    row = {**base}
    for col in c:
        if col.startswith('loc_'):
            row[col] = 1 if col == f'loc_{loc}' else 0
    
    df = pd.DataFrame([row])[c]
    pred = m.predict(df)[0]
    print(f'{loc:12s} -> {pred}')