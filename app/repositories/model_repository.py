import os
import joblib

# Project root folder (3 levels up from app/repositories/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "models", "house_model.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "models", "house_imputer.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "house_columns.pkl")


class ModelRepository:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.columns = None
        self.loaded = False

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file nahi mila: {MODEL_PATH}\n"
                "Pehle 'python scripts/train.py' chalao."
            )
        
        self.model = joblib.load(MODEL_PATH)
        self.imputer = joblib.load(IMPUTER_PATH)
        self.columns = joblib.load(COLUMNS_PATH)
        self.loaded = True
        print(f"[ModelRepository] Loaded model with {len(self.columns)} features.")


# Global instance
model_repository = ModelRepository()