from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///inference.db", connect_args={"check_same_thread": False})
Base = declarative_base()


class InferenceJob(Base):
    __tablename__ = "inference_jobs"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    n_jets            = Column(Integer, nullable=False)
    latency_ms        = Column(Float,   nullable=False)
    throughput_jets_s = Column(Float,   nullable=False)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id         = Column(Integer, primary_key=True)
    name       = Column(String,  nullable=False)
    version    = Column(String,  nullable=False)
    auc_roc    = Column(Float)
    bg_rej_30  = Column(Float)
    bg_rej_50  = Column(Float)
    trained_at = Column(DateTime)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    batch_size        = Column(Integer,  nullable=False)
    n_jets            = Column(Integer,  nullable=False)
    p50_chunk_ms      = Column(Float)
    p95_chunk_ms      = Column(Float)
    p99_chunk_ms      = Column(Float)
    throughput_jets_s = Column(Float)


SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)