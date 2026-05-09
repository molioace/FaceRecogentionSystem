from fastapi import FastAPI
from app.routers import recognition

app = FastAPI()

app.include_router(recognition.router, prefix="/recognition", tags=["recognition"])
