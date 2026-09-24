"""
main.py
-------
App entrypoint. Loads model at startup and mounts routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.predict import router as predict_router
from app.models.model_loader import model_store

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="4-Layer Architecture: routes -> schemas -> services -> models",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    """Load model artifacts when server starts."""
    model_store.load()


@app.get("/", tags=["Root"])
def root():
    return {"message": "House Price Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model_store.loaded}


app.include_router(predict_router, tags=["Prediction"])