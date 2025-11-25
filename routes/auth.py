from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from email_utils import send_verification_email

import auth
import crud
import schemas
from db import get_db

router = APIRouter()

@router.post("/signup")
async def create_user(user: schemas.UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        db_user = crud.get_user_by_email(db, email=user.email)
        if db_user and db_user.is_email_verified:

            raise HTTPException(
                status_code=400,
                detail="Email already registered! Please log in."
            )
        
        if db_user and not db_user.is_email_verified:
            raise HTTPException(
                status_code=400, 
                detail="Email not verified, please check your email for verification."
            )
        
        new_user = crud.create_user(db=db, user=user)
        verification_link = f"http://localhost:8000/auth/verify-email?token={new_user.email_verification_token}"
        background_tasks.add_task(send_verification_email, [new_user.email], new_user.username, verification_link)
        
        return { "message": "Please check your email to verify your account." }
    except Exception as e:
        print(f"Error while creating user: {e}")
        if(not isinstance(e, HTTPException)):
            raise HTTPException(
                status_code=500, 
                detail="Internal Server Error"
            )

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    try:
        user = crud.get_user_by_email_verification_token(db, token=token)
        if not user:
            raise HTTPException(
                status_code=400, 
                detail="Invalid verification token"
            )
        
        crud.verify_user_email(db, user=user)
        return {"message": "Email verified successfully. You can now log in."}
    except Exception as e:
        print(f"Error while verifying email: {e}")
        if(not isinstance(e, HTTPException)):
            raise HTTPException(
                status_code=500, 
                detail="Internal Server Error"
            )


@router.post("/login", response_model=schemas.Token)
async def login_for_access_token(login_request: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=login_request.email)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Email is not registered, Please sign up first!",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not auth.verify_password(login_request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password!",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_email_verified:
        raise HTTPException(
            status_code=400, 
            detail="Email not verified. Please check your email for a verification."
        )
    
    access_token = auth.create_access_token(
        data={"sub": user.email}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.User)
async def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    return current_user