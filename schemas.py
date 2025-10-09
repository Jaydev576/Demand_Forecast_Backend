from typing import Optional
from pydantic import BaseModel

class UserBase(BaseModel):
    username: str | None = None
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_email_verified: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class LoginRequest(BaseModel):
    email: str
    password: str

class UploadBase(BaseModel):
    filename: str
    key: str
    bucket: str
    size_bytes: int
    content_type: str

class UploadCreate(UploadBase):
    user_id: int

class UploadCompleteRequest(BaseModel):
    upload_id: int
    s3_key: str

class TrainResponse(BaseModel):
    message: str
    model_type: str
    metrics: dict
    sample_fig_json: dict

class PredictRequest(BaseModel):
    product_category: str
    product: str
    city: str
    num_days: int = 30
    price: Optional[float] = None
    discount: Optional[float] = None