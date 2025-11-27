from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from utils.auth import get_current_active_user
from db.db import get_db
from models.models import User
import services.crud as crud

router = APIRouter()

@router.get("/features")
def get_features(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """
    Extract distinct features (column names) from csv uploaded by user 
    """
    features = crud.get_distinct_features(db, user_id=current_user.id)
    if not features:
        return {"data": [], "message": "No features found."}
    return features
