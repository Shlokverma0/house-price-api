"""
test_api.py
-----------
Tests for the House Price Prediction API.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict():
    payload = {
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Price_per_SqFt": 8000,
        "Year_Built": 2015,
        "Parking_Space": 1,
        "location": "Mumbai"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price_in_lakhs" in response.json()