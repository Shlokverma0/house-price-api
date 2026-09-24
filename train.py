"""
train.py
--------
Downloads the Indian House Price dataset, preprocesses location (City),
trains a Multiple Linear Regression model, and saves artifacts.
"""

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "Price_in_Lakhs"


def load_dataset() -> pd.DataFrame:
    import kagglehub

    dataset_path = kagglehub.dataset_download("srisyra02/house-price-prediction-dataset")
    csv_path = f"{dataset_path}/indian_house_price_prediction_data.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def main():
    # ---------- Step 1: Load ----------
    df = load_dataset()

    # ---------- Step 2: Location + Yes/No Preprocessing ----------
    if 'City' not in df.columns:
        raise KeyError("Dataset me 'City' column nahi mila.")

    df['location'] = df['City']

    # Yes/No -> 1/0 convert karo (Parking_Space)
    df['Parking_Space'] = df['Parking_Space'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Top 50 cities rakho, baaki 'Other'
    top_cities = df['location'].value_counts().nlargest(50).index
    df['location'] = df['location'].apply(lambda x: x if x in top_cities else 'Other')

    # One-hot encoding
    df = pd.get_dummies(df, columns=['location'], prefix='loc', drop_first=False)
    location_cols = [col for col in df.columns if col.startswith('loc_')]

    # ---------- Step 3: Features ----------
    base_features = ['BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built', 'Parking_Space']

    for col in base_features:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' dataset me nahi mila.")

    FEATURES = base_features + location_cols
    print(f"\nTotal Features ({len(FEATURES)})")

    df = df[FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])

    # ---------- Step 4: Split ----------
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Step 5: Impute ----------
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    # ---------- Step 6: Train ----------
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    # ---------- Step 7: Evaluate ----------
    preds = model.predict(X_test_imputed)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"\nMAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.3f}")

    # ---------- Step 8: Save ----------
    clean_df = pd.concat(
        [pd.concat([X_train_imputed, X_test_imputed]), pd.concat([y_train, y_test])],
        axis=1,
    )
    clean_df.to_csv("house_data_clean.csv", index=False)

    joblib.dump(model, "house_model.pkl")
    joblib.dump(FEATURES, "house_columns.pkl")
    joblib.dump(imputer, "house_imputer.pkl")

    print("\nSaved: house_model.pkl, house_columns.pkl, house_imputer.pkl, house_data_clean.csv")


if __name__ == "__main__":
    main()