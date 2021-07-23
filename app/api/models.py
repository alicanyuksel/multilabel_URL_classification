from pydantic import BaseModel
from config.config import API_NAME
import datetime


class TextInput(BaseModel):
    url: str


class TextOutput(BaseModel):
    url: str
    parsed_url: str
    predictions: list
    inference_time: datetime.timedelta


class Welcome(BaseModel):
    msg = f"Welcome to {API_NAME}"
