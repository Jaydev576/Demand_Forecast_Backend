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