"""
predict.py
----------
Layer 1: API routes — receives request, calls service, returns response.
"""

from fastapi import APIRouter
from app.schemas.house import HouseFeatures, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    """Predict house price based on input features."""
    price = service.predict(features)
    return PredictionResponse(
        predicted_price_in_lakhs=round(price, 2),
        currency="INR"
    )