"""
house.py
--------
Layer 2: Pydantic schemas for input validation and response formatting.
"""
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class LocationEnum(str, Enum):
    ahmedabad = "Ahmedabad"
    amritsar = "Amritsar"
    bangalore = "Bangalore"
    bhopal = "Bhopal"
    bhubaneswar = "Bhubaneswar"
    bilaspur = "Bilaspur"
    chennai = "Chennai"
    coimbatore = "Coimbatore"
    cuttack = "Cuttack"
    dehradun = "Dehradun"
    durgapur = "Durgapur"
    dwarka = "Dwarka"
    faridabad = "Faridabad"
    gaya = "Gaya"
    gurgaon = "Gurgaon"
    guwahati = "Guwahati"
    haridwar = "Haridwar"
    hyderabad = "Hyderabad"
    indore = "Indore"
    jaipur = "Jaipur"
    jamshedpur = "Jamshedpur"
    jodhpur = "Jodhpur"
    kochi = "Kochi"
    kolkata = "Kolkata"
    lucknow = "Lucknow"
    ludhiana = "Ludhiana"
    mangalore = "Mangalore"
    mumbai = "Mumbai"
    mysore = "Mysore"
    nagpur = "Nagpur"
    new_delhi = "New Delhi"
    noida = "Noida"
    patna = "Patna"
    pune = "Pune"
    raipur = "Raipur"
    ranchi = "Ranchi"
    silchar = "Silchar"
    surat = "Surat"
    trivandrum = "Trivandrum"
    vijayawada = "Vijayawada"
    visakhapatnam = "Vishakhapatnam"
    warangal = "Warangal"


class HouseFeatures(BaseModel):
    BHK: int = Field(..., ge=1, le=10)
    Size_in_SqFt: float = Field(..., ge=100, le=20000)
    Price_per_SqFt: float = Field(..., ge=500, le=50000)
    Year_Built: int = Field(..., ge=1900, le=2026)
    Parking_Space: int = Field(..., ge=0, le=1)
    location: LocationEnum = Field(..., description="City name")

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, v):
        if isinstance(v, str):
            synonyms = {
                "Delhi": "New Delhi",
                "Bombay": "Mumbai",
                "Bengaluru": "Bangalore",
                "Madras": "Chennai",
                "Calcutta": "Kolkata",
                "Gurugram": "Gurgaon",
            }
            return synonyms.get(v.strip(), v.strip())
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "BHK": 3,
                "Size_in_SqFt": 1500,
                "Price_per_SqFt": 5000,
                "Year_Built": 2015,
                "Parking_Space": 1,
                "location": "New Delhi",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_in_lakhs: float
    currency: str = "INR"