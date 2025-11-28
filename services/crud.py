from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
import models.models as models
import schemas.schemas as schemas
from core.security import get_password_hash
import secrets

# -----------------------
# Users / Upload helpers
# -----------------------
def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    hashed_password = get_password_hash(user.password)
    verification_token = secrets.token_urlsafe(32)
    db_user = models.User(
        username=user.username,
        email=user.email,
        password=hashed_password,
        email_verification_token=verification_token
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email_verification_token(db: Session, token: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email_verification_token == token).first()

def verify_user_email(db: Session, user: models.User) -> models.User:
    user.is_email_verified = True
    user.email_verification_token = None
    db.commit()
    db.refresh(user)
    return user

def create_upload(db: Session, upload: schemas.UploadCreate) -> models.Upload:
    db_upload = models.Upload(
        user_id=upload.user_id,
        filename=upload.filename,
        key=upload.key,
        bucket=upload.bucket,
        size_bytes=upload.size_bytes,
        content_type=upload.content_type
    )
    db.add(db_upload)
    db.commit()
    db.refresh(db_upload)
    return db_upload

def get_upload_by_userid(db: Session, user_id: int) -> List[models.Upload]:
    return db.query(models.Upload).filter(models.Upload.user_id == user_id).all()

def get_upload_by_id(db: Session, upload_id: int) -> Optional[models.Upload]:
    return db.query(models.Upload).filter(models.Upload.id == upload_id).first()

def get_upload_by_key(db: Session, key: str) -> Optional[models.Upload]:
    return db.query(models.Upload).filter(models.Upload.key == key).first()

def list_uploads_for_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Upload]:
    return db.query(models.Upload).filter(models.Upload.user_id == user_id).offset(skip).limit(limit).all()


# -----------------------
# Upload helpers
# -----------------------
def get_upload_s3_key(db: Session, upload_id: int) -> Optional[str]:
    upload = db.query(models.Upload).filter(models.Upload.id == upload_id).first()
    if upload:
        return upload.key
    return None

def get_userid_by_upload(db: Session, upload_id: int) -> Optional[int]:
    upload = db.query(models.Upload).filter(models.Upload.id == upload_id).first()
    if upload:
        return upload.user_id
    return None

# -----------------------
# Forecast helpers
# -----------------------
def save_forecast(
    db: Session,
    *,
    upload_id: Optional[int],
    user_id: Optional[int],
    product_category: Optional[str],
    product: str,
    city: str,
    start_date: datetime,
    end_date: datetime,
    num_days: int,
    predictions: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> models.Forecast:
    """
    Persist a forecast run into forecasts table.
    predictions -> list of {"date": "YYYY-MM-DD", "predicted_quantity_sold": int}
    params -> request parameters / overrides
    """
    rec = models.Forecast(
        upload_id = upload_id,
        user_id = user_id,
        product_category = product_category,
        product = product,
        city = city,
        start_date = start_date,
        end_date = end_date,
        num_days = num_days,
        params = params or {},
        predictions = predictions,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

def get_forecast_by_id(db: Session, forecast_id: int) -> Optional[models.Forecast]:
    return db.query(models.Forecast).filter(models.Forecast.id == forecast_id).first()

def query_forecasts(
    db: Session,
    *,
    product: Optional[str] = None,
    city: Optional[str] = None,
    product_category: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    user_id: Optional[int] = None,
    upload_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
) -> List[models.Forecast]:
    q = db.query(models.Forecast)
    if product:
        q = q.filter(models.Forecast.product == product)
    if city:
        q = q.filter(models.Forecast.city == city)
    if product_category:
        q = q.filter(models.Forecast.product_category == product_category)
    if user_id:
        q = q.filter(models.Forecast.user_id == user_id)
    if upload_id:
        q = q.filter(models.Forecast.upload_id == upload_id)
    if date_from:
        q = q.filter(models.Forecast.start_date >= date_from)
    if date_to:
        q = q.filter(models.Forecast.end_date <= date_to)
    q = q.order_by(models.Forecast.created_at.desc()).offset(offset).limit(limit)
    return q.all()

def delete_forecast(db: Session, forecast_id: int) -> bool:
    rec = db.query(models.Forecast).filter(models.Forecast.id == forecast_id).first()
    if not rec:
        return False
    db.delete(rec)
    db.commit()
    return True

# -----------------------
# TrainingRun helpers
# -----------------------
def create_training_run(
    db: Session,
    *,
    upload_id: Optional[int] = None,
    user_id: Optional[int] = None,
    status: str = "processing",
    model_path: Optional[str] = None,
    model_meta_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    features: Optional[List[str]] = None,
    rows_trained: Optional[int] = None
) -> models.TrainingRun:
    rec = models.TrainingRun(
        upload_id = upload_id,
        user_id = user_id,
        status = status,
        model_path = model_path,
        model_meta_path = model_meta_path,
        metrics = metrics or {},
        features = features or [],
        rows_trained = rows_trained,
        started_at = datetime.utcnow()
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

def get_training_run(db: Session, run_id: int) -> Optional[models.TrainingRun]:
    return db.query(models.TrainingRun).filter(models.TrainingRun.id == run_id).first()

def list_training_runs(
    db: Session,
    *,
    upload_id: Optional[int] = None,
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[models.TrainingRun]:
    q = db.query(models.TrainingRun)
    if upload_id:
        q = q.filter(models.TrainingRun.upload_id == upload_id)
    if user_id:
        q = q.filter(models.TrainingRun.user_id == user_id)
    if status:
        q = q.filter(models.TrainingRun.status == status)
    q = q.order_by(models.TrainingRun.started_at.desc()).offset(offset).limit(limit)
    return q.all()

def update_training_run(
    db: Session,
    run_id: int,
    *,
    status: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    model_meta_path: Optional[str] = None,
    finished_at: Optional[datetime] = None,
    error_message: Optional[str] = None,
    rows_trained: Optional[int] = None
) -> Optional[models.TrainingRun]:
    rec = db.query(models.TrainingRun).filter(models.TrainingRun.id == run_id).first()
    if not rec:
        return None
    if status is not None:
        rec.status = status
    if metrics is not None:
        rec.metrics = metrics
    if model_path is not None:
        rec.model_path = model_path
    if model_meta_path is not None:
        rec.model_meta_path = model_meta_path
    if finished_at is not None:
        rec.finished_at = finished_at
    if error_message is not None:
        rec.error_message = error_message
    if rows_trained is not None:
        rec.rows_trained = rows_trained
    db.commit()
    db.refresh(rec)
    return rec

def fail_training_run(db: Session, run_id: int, error_message: str) -> Optional[models.TrainingRun]:
    return update_training_run(db, run_id, status="failed", error_message=error_message, finished_at=datetime.utcnow())

# -----------------------
# Misc helpers
# -----------------------
def get_forecasts_count(db: Session) -> int:
    return db.query(models.Forecast).count()

def get_training_runs_count(db: Session) -> int:
    return db.query(models.TrainingRun).count()

def get_distinct_features(db: Session, user_id: int) -> Optional[models.DistinctFeature]:
    return (
        db.query(models.DistinctFeature)
        .filter(models.DistinctFeature.user_id == user_id)
        .order_by(models.DistinctFeature.id.desc())
        .first()
    )

def create_business_insight(db: Session, insight: schemas.BusinessInsightCreate) -> models.BusinessInsight:
    db_insight = models.BusinessInsight(
        user_id=insight.user_id,
        kpis=insight.kpis,
        charts=insight.charts,
    )
    db.add(db_insight)
    db.commit()
    db.refresh(db_insight)
    return db_insight

def get_latest_business_insight(db: Session, user_id: int) -> Optional[models.BusinessInsight]:
    return (
        db.query(models.BusinessInsight)
        .filter(models.BusinessInsight.user_id == user_id)
        .order_by(models.BusinessInsight.created_at.desc())
        .first()
    )

def list_forecasts_for_user(db: Session, user_id: int) -> List[models.Forecast]:
    return db.query(models.Forecast).filter(models.Forecast.user_id == user_id).order_by(models.Forecast.created_at.desc()).all()
