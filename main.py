from fastapi import FastAPI
from db.db import engine
import models.models as models
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, upload, user, feature, insights, predict

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

app.include_router(auth.router, prefix="/auth")
app.include_router(upload.router, prefix="/upload")
app.include_router(predict.router, prefix="/predict")
app.include_router(user.router, prefix="/user")
app.include_router(feature.router, prefix="/feature")
app.include_router(insights.router, prefix="/insights")


@app.get("/")
def read_root():
    return {"massage": "FastAPI is running!!"}
