from sqlalchemy.orm import Session
import models, schemas
from security import get_password_hash
import secrets

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
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

def get_user_by_email_verification_token(db: Session, token: str):
    return db.query(models.User).filter(models.User.email_verification_token == token).first()

def verify_user_email(db: Session, user: models.User):
    user.is_email_verified = True
    user.email_verification_token = None
    db.commit()
    db.refresh(user)
    return user


def create_upload(db: Session, upload: schemas.UploadCreate):
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