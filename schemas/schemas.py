from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# -----------------------
# User related schemas
# -----------------------
class UserBase(BaseModel):
    username: Optional[str] = None
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_email_verified: bool = False

    class Config:
        # orm_mode allows returning SQLAlchemy models directly in responses
        orm_mode = True
        # If you are using pydantic v2, consider using:
        # model_config = {"from_attributes": True}
        # or keep both in a dual-support environment.


# -----------------------
# Auth / Token schemas
# -----------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str


# -----------------------
# Upload schemas
# -----------------------
class UploadBase(BaseModel):
    filename: str
    key: str
    bucket: str
    size_bytes: int
    content_type: str

class UploadCreate(UploadBase):
    user_id: int

class UploadOut(UploadBase):
    id: int
    user_id: Optional[int] = None
    uploaded_at: Optional[str] = None

    class Config:
        orm_mode = True

class UploadCompleteRequest(BaseModel):
    upload_id: int
    s3_key: Optional[str] = None


# -----------------------
# Training / Prediction schemas
# -----------------------
class TrainResponse(BaseModel):
    message: str
    model_type: str
    metrics: Dict[str, Any]
    sample_fig_json: Dict[str, Any]
    training_run_id: Optional[int] = None

class PredictRequest(BaseModel):
    product_category: str
    product: str
    city: str
    num_days: int = Field(30, ge=1, le=365)  # validate reasonable horizon
    price: Optional[float] = None
    discount: Optional[float] = None

class PredictionRecord(BaseModel):
    date: str
    predicted_quantity_sold: int

class PredictResponse(BaseModel):
    product: str
    city: str
    num_days: int
    predictions: list[PredictionRecord]
    figure_json: Optional[Dict[str, Any]] = None
    feature_importance_json: Optional[Dict[str, Any]] = None
    forecast_id: Optional[int] = None

    class Config:
        orm_mode = True

class Forecast(BaseModel):
    id: int
    product_category: str
    product: str
    city: str
    num_days: int
    predictions: list[PredictionRecord]
    created_at: datetime

    class Config:
        orm_mode = True

class ForecastOut(BaseModel):
    forecasts: list[Forecast]

class BusinessInsight(BaseModel):
    id: int
    user_id: int
    kpis: Dict[str, Any]
    charts: Dict[str, Any]
    # created_at: datetime

    class Config:
        orm_mode = True

class BusinessInsightCreate(BaseModel):
    user_id: int
    kpis: Dict[str, Any]
    charts: Dict[str, Any]
