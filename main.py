from fastapi import FastAPI
from db import engine
import models

# models.Base.metadata.drop_all(bind=engine) # to drop all tables in db
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

from routers import users, uploads

app.include_router(users.router, prefix="/user")
# app.include_router(uploads.router, prefix="/upload")

@app.get("/test")
def read_root():
    return {"check": "Hello World!"}
