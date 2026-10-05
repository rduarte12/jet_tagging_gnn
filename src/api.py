from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, HTTPException
from src.engine import Inferencer
from src.db import init_db
import logging
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from src.db import SessionLocal, InferenceLog, AvailableModel

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()

    logger.info("Loading model...")
    app.state.inferencer = Inferencer("models/EdgeConv_best.pt")
    
    logger.info("Model ready. Server is up.")
    
    yield
    
    logger.info("Shutting down.")

app = FastAPI(lifespan=lifespan)

# schemas pydantic
# 1 - request schemas
class PredictRequest(BaseModel):
    particles: list[list[float]] = Field(...,
                                                min_length=1, 
                                                max_length=200, 
    description="One jet for each request, with up to 200 particles. Each particle is represented by a list of 4 floats")

class BatchRequest(BaseModel):
    jets: list[list[list[float]]] = Field(...,
                                          min_length=1,
                                          max_length=100,
    description="A batch of jets, each with up to 200 particles. Each particle is represented by a list of 4 floats")

# 2 - response schemas
class PredictResponse(BaseModel):
    score: float = Field(..., description="The predicted score for the jet")
    prediction: int = Field(..., description="The predicted class for the jet (0 or 1)")
    latency_ms: float = Field(..., description="The time taken to make the prediction in milliseconds")
    job_id: int | None = Field(..., description="The ID of the inference job in the database")

class BatchResponse(BaseModel):
    results: list[PredictResponse] = Field(..., description="The list of predictions for each jet in the batch")
    throughput: int = Field(..., description="The number of jets processed per second")
    total_ms: float = Field(..., description="The total time taken to process the batch in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="The health status of the API")
    model_ready: bool = Field(..., description="Whether the model is loaded and ready for inference")

class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="The ID of the model in the database")
    name: str = Field(..., description="The name of the model")
    auc_roc: float | None = Field(None, description="AUC-ROC score")

# api endpoints and necessary dependencies

# database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# inferencer dependency
def get_inferencer(request: Request) -> Inferencer:
    return request.app.state.inferencer


# endpoints
# health check endpoint
@app.get("/health", response_model=HealthResponse)
def health(inferencer: Inferencer = Depends(get_inferencer)):
    return HealthResponse(
        status="ok",
        model_ready=inferencer.is_ready()
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest,
            inferencer: Inferencer = Depends(get_inferencer),
            db: Session = Depends(get_db)):

    if not inferencer.is_ready():
        raise HTTPException(status_code=503, detail="Model is not ready for inference.")