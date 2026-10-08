from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, HTTPException
from src.engine import Inferencer
from src.db import init_db
import logging
from typing import cast
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from src.db import SessionLocal, InferenceJob, ModelRegistry, BenchmarkRun
import numpy as np
from datetime import datetime, timezone
import json, os
from src.batch import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()

    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        db = SessionLocal()
        try:
            if not db.query(ModelRegistry).filter_by(id=1).first():
                db.add(ModelRegistry(
                    id=1,
                    name=m.get("model_name", "EdgeConv"),
                    version=m.get("version", "1.0"),
                    auc_roc=m.get("auc_roc"),
                    bg_rej_30=m.get("bg_rejection_30"),
                    bg_rej_50=m.get("bg_rejection_50"),
                    trained_at=datetime.now(timezone.utc),
                ))
                db.commit()
                logger.info("Model registered in model_registry.")
        finally:
            db.close()

    logger.info("Loading model...")
    app.state.inferencer = Inferencer("models/EdgeConv_best.pt")
    logger.info("Model ready. Server is up.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Jet Tagging GNN",
    description="EdgeConv based top quark jet classifier. HEP benchmark dataset. /n Developed by Rafael Duarte /n E-mail: rmduarte@usp.br",
    lifespan=lifespan,
)


# Schemas

class BatchRequest(BaseModel):
    jets: list[list[list[float]]] = Field(
        ..., min_length=1, max_length=100,
        description="List of jets. Each jet is a list of particles [pT, η, φ, E]. Max 200 particles per jet.",
    )

class BatchRunRequest(BaseModel):
    jets: list[list[list[float]]] = Field(..., min_length=1)
    batch_size: int = Field(256, ge=1, le=1024)

class PredictResult(BaseModel):
    score: float
    prediction: int
    latency_ms: float

class BatchResponse(BaseModel):
    results: list[PredictResult]
    n_jets: int
    throughput_jets_s: float
    total_ms: float

class HealthResponse(BaseModel):
    status: str
    model_ready: bool

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    version: str
    auc_roc: float | None
    bg_rej_30: float | None
    bg_rej_50: float | None


# Dependencies 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_inferencer(request: Request) -> Inferencer:
    return request.app.state.inferencer


# Endpoints

@app.get("/health", response_model=HealthResponse)
def health(inferencer: Inferencer = Depends(get_inferencer)):
    return HealthResponse(status="ok", model_ready=inferencer.is_ready())


@app.get("/model", response_model=ModelInfoResponse)
def model_info(db: Session = Depends(get_db)):
    """Returns the metadata of the model currently loaded (from metrics.json at startup)."""
    record = db.query(ModelRegistry).filter_by(id=1).first()
    if not record:
        raise HTTPException(status_code=404, detail="No model registered.")
    return record


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    req: BatchRequest,
    inferencer: Inferencer = Depends(get_inferencer),
    db: Session = Depends(get_db),
):
    """Classify N jets. Returns per-jet scores and batch throughput."""
    if not inferencer.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    raws = []
    for jet in req.jets:
        raw = np.zeros((200, 4), dtype=np.float32)
        raw[:len(jet)] = np.array(jet, dtype=np.float32)
        raws.append(raw)

    batch_result = inferencer.predict_batch(raws)

    db.add(InferenceJob(
        n_jets=len(req.jets),
        latency_ms=batch_result["total_ms"],
        throughput_jets_s=batch_result["throughput"],
    ))
    db.commit()

    return BatchResponse(
        results=[PredictResult(**r) for r in batch_result["results"]],
        n_jets=len(req.jets),
        throughput_jets_s=batch_result["throughput"],
        total_ms=batch_result["total_ms"],
    )


@app.post("/batch/run")
def batch_run(
    req: BatchRunRequest,
    inferencer: Inferencer = Depends(get_inferencer),
    db: Session = Depends(get_db),
):
    """Benchmark: process jets in fixed-size chunks, return p50/p95/p99 latency and throughput."""
    if not inferencer.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    raws = []
    for jet in req.jets:
        raw = np.zeros((200, 4), dtype=np.float32)
        raw[:len(jet)] = np.array(jet, dtype=np.float32)
        raws.append(raw)

    stats = run_benchmark(raws, req.batch_size, inferencer)

    db.add(BenchmarkRun(
        batch_size=stats["batch_size"],
        n_jets=stats["n_jets"],
        p50_chunk_ms=stats["p50_chunk_ms"],
        p95_chunk_ms=stats["p95_chunk_ms"],
        p99_chunk_ms=stats["p99_chunk_ms"],
        throughput_jets_s=stats["throughput_jets_s"],
    ))
    db.commit()

    return stats