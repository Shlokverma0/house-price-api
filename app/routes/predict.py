from fastapi import APIRouter
from app.schemas.house import HouseFeatures, PredictionResponse
from app.controllers.prediction_controller import PredictionController

router = APIRouter()
controller = PredictionController()


@router.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    """Predict house price based on input features."""
    return controller.predict_price(features)