from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import auth
import crud
from db import get_db

router = APIRouter()

@router.get("/models/status")
def get_mode_status(db: Session = Depends(get_db), user = Depends(auth.get_current_active_user)):
    
    training_models = crud.list_training_runs(db, user_id=user.id)
    return { "data" : training_models }