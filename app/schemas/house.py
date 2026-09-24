"""
house.py
--------
Layer 2: Pydantic schemas for input validation and response formatting.
"""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    BHK: int = Field(..., gt=0, description="Number of Bedrooms, Hall, Kitchen")
    Size_in_SqFt: float = Field(..., gt=0, description="Size of property in square feet")
    Price_per_SqFt: float = Field(..., gt=0, description="Price per square feet")
    Year_Built: int = Field(..., ge=1900, le=2026, description="Year property was built")
    Parking_Space: int = Field(..., ge=0, le=1, description="1 for Yes, 0 for No")
    location: str = Field(..., description="City name (e.g., Mumbai, Bangalore, New Delhi)")


class PredictionResponse(BaseModel):
    predicted_price_in_lakhs: float
    currency: str = "INR"
    note: str = "Price is in Lakhs (1 Lakh = 100,000 INR)"