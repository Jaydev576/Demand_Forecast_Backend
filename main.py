from fastapi import FastAPI
from database import engine
import models

# models.Base.metadata.drop_all(bind=engine) # to drop all tables in db
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

from routers import users

app.include_router(users.router, prefix="/v1/user")

@app.get("/v1")
def read_root():
    return {"message": "Hello World!"}
