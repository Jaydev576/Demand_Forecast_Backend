from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from auth import get_current_active_user
from db import get_db
from models import User
import crud

router = APIRouter()

@router.get("/features")
def get_features(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    features = crud.get_distinct_features(db, user_id=current_user.id)
    if not features:
        return {"column_names": [], "product": [], "category": [], "city": []}
    return features
