from sqlalchemy import Column, Integer, String, Boolean, BigInteger, DateTime, func, ForeignKey
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
