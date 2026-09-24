from app.schemas.house import HouseFeatures, PredictionResponse
from app.services.prediction_service import PredictionService


class PredictionController:
    def __init__(self):
        self.service = PredictionService()

    def predict_price(self, features: HouseFeatures) -> PredictionResponse:
        price = self.service.predict(features)
        return PredictionResponse(
            predicted_price_in_lakhs=round(price, 2),
            currency="INR"
        )