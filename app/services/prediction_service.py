import pandas as pd
from app.schemas.house import HouseFeatures
from app.repositories.model_repository import model_repository


class PredictionService:
    def predict(self, features: HouseFeatures) -> float:
        if not model_repository.loaded:
            model_repository.load()

        data = features.model_dump()
        location = data.pop("location").strip().title()

        synonyms = {
            "Delhi": "New Delhi",
            "Bombay": "Mumbai",
            "Bengaluru": "Bangalore",
            "Madras": "Chennai",
            "Calcutta": "Kolkata",
            "Gurugram": "Gurgaon",
        }
        location = synonyms.get(location, location)

        for col in model_repository.columns:
            if col.startswith("loc_"):
                data[col] = 0

        loc_col = f"loc_{location}"
        if loc_col in data:
            data[loc_col] = 1
        elif "loc_Other" in data:
            data["loc_Other"] = 1

        df = pd.DataFrame([data])
        df = df[model_repository.columns]

        df_imputed = model_repository.imputer.transform(df)
        prediction = model_repository.model.predict(df_imputed)[0]
        return float(prediction)