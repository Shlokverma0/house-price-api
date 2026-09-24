"""
model_loader.py
---------------
Layer 4: Loads the trained model, imputer, and column list at startup.
"""

import os
import joblib

# Project root folder ka path nikaalo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "house_model.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "house_imputer.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "house_columns.pkl")

class ModelStore:
    """Holds the model, imputer, and column list in memory."""
    
    def __init__(self):
        self.model = None
        self.imputer = None
        self.columns = None
        self.loaded = False

    def load(self):
        """Load all artifacts from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file nahi mila: {MODEL_PATH}\n"
                "Pehle 'python train.py' chalao."
            )
        
        self.model = joblib.load(MODEL_PATH)
        self.imputer = joblib.load(IMPUTER_PATH)
        self.columns = joblib.load(COLUMNS_PATH)
        self.loaded = True
        print(f"[ModelStore] Loaded model with {len(self.columns)} features.")

# Global instance - main.py isko import karega
model_store = ModelStore()