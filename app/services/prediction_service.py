"""
prediction_service.py
---------------------
Layer 3: Business logic — takes validated input, prepares features,
calls the model, and returns a prediction.
"""

import pandas as pd
from app.schemas.house import HouseFeatures
from app.models.model_loader import model_store

class PredictionService:
    """Handles feature engineering and model prediction."""

    def predict(self, features: HouseFeatures) -> float:
        if not model_store.loaded:
            model_store.load()

        # Input ko dict me convert karo
        data = features.model_dump()

        # Location ko alag nikaalo aur normalize karo
        location = data.pop("location").strip().title()

        # Synonym mapping: user-friendly naam ko dataset ke naam se match karo
        synonyms = {
            "Delhi": "New Delhi",
            "Bombay": "Mumbai",
            "Bengaluru": "Bangalore",
            "Madras": "Chennai",
            "Calcutta": "Kolkata",
            "Gurugram": "Gurgaon",
        }
        location = synonyms.get(location, location)

        # Saare location columns ko 0 se initialize karo
        for col in model_store.columns:
            if col.startswith("loc_"):
                data[col] = 0

        # Matching location column ko 1 set karo
        loc_col = f"loc_{location}"
        if loc_col in data:
            data[loc_col] = 1
        elif "loc_Other" in data:
            data["loc_Other"] = 1

        # DataFrame banao aur columns ka exact order set karo
        df = pd.DataFrame([data])
        df = df[model_store.columns]

        # Imputer lagao
        df_imputed = model_store.imputer.transform(df)

        # Prediction karo
        prediction = model_store.model.predict(df_imputed)[0]
        return float(prediction)