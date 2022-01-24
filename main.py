from fastapi import FastAPI
from pydantic import BaseModel
from NBC import toBook

app = FastAPI()

class Data(BaseModel):
    values: list


@app.get("/")
def read_root(message: str):
    return toBook(message)