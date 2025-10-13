from sqlalchemy import Column, Integer, String, Boolean, BigInteger, DateTime, func, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String, index=True)
    is_email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String, index=True, unique=True)


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    filename = Column(String(255))
    key = Column(String(512), unique=True, index=True)
    bucket = Column(String(128))
    size_bytes = Column(BigInteger)
    content_type = Column(String(128))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())


class Forecast(Base):
    """
    Stores forecast runs / results.
    - predictions: JSONB list of {"date": "...", "predicted_quantity_sold": ...}
    - params: JSONB of request parameters (price/discount/overrides)
    """
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), index=True, nullable=True)   # optional link to upload
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)       # who requested the forecast
    product_category = Column(String, index=True, nullable=True)
    product = Column(String, index=True, nullable=True)
    city = Column(String, index=True, nullable=True)
    start_date = Column(DateTime(timezone=True), index=True, nullable=False)
    end_date = Column(DateTime(timezone=True), index=True, nullable=False)
    num_days = Column(Integer, nullable=False)
    params = Column(JSONB, nullable=True)
    predictions = Column(JSONB, nullable=False)          # list of dicts
    figure_json = Column(Text, nullable=True)            # plotly figure JSON string
    feature_importance_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index('ix_forecasts_product_city_dates', 'product', 'city', 'start_date', 'end_date'),
    )


class TrainingRun(Base):
    """
    Stores metadata about training runs.
    - status: "processing", "completed", "failed"
    - metrics: JSONB with mae/rmse/r2 etc
    """
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), index=True, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    status = Column(String, index=True, nullable=False, default="processing")  # processing / completed / failed
    model_path = Column(String, nullable=True)         # path (local or s3)
    model_meta_path = Column(String, nullable=True)    # metadata/artifacts path
    metrics = Column(JSONB, nullable=True)             # e.g. {"mae":..,"rmse":..}
    features = Column(JSONB, nullable=True)            # feature list
    rows_trained = Column(Integer, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index('ix_trainingruns_upload_status', 'upload_id', 'status'),
    )
