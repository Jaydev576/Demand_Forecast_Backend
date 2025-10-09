from fastapi import FastAPI
from db import engine
import models
from fastapi.middleware.cors import CORSMiddleware
from routes import users, uploads, train

# models.Base.metadata.drop_all(bind=engine) # to drop all tables in db
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Define the list of allowed origins
origins = [
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://localhost:5173", 
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies/authorization headers
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers
)

app.include_router(users.router, prefix="/user")
app.include_router(uploads.router, prefix="/upload")
app.include_router(train.router, prefix="/train")

@app.get("/api")
def read_root():
    return {"massage": "Welcome to backend!!"}
