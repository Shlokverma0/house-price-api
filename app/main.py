from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Yahan 'routes' kar do (pehle 'routers' tha)
from app.routes.predict import router as predict_router
from app.repositories.model_repository import model_repository

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="4-Layer Architecture (Controller-Service-Repository)",
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
    model_repository.load()

@app.get("/", tags=["Root"])
def root():
    return {"message": "House Price Prediction API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model_repository.loaded}

app.include_router(predict_router, tags=["Prediction"])