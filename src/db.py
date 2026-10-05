from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///inference.db", connect_args={"check_same_thread": False})

Base = declarative_base()


class InferenceLog(Base):
    __tablename__ = "inference_log"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    model_id   = Column(Integer, ForeignKey("available_models.id"), nullable=False)
    score      = Column(Float,   nullable=False)
    prediction = Column(Integer, nullable=False)
    latency_ms = Column(Float,   nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class AvailableModel(Base):
    __tablename__ = "available_models"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String,  nullable=False)
    path          = Column(String,  nullable=False)
    auc_roc       = Column(Float)
    registered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

SessionLocal = sessionmaker(bind=engine)