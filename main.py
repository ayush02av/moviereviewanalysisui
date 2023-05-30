from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import model

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/ping")
async def ping():
    return "pong"

@app.get("/predict/{review}")
async def predict(review: str):
    try:
        return model.predict(review)
    except:
        raise HTTPException(400)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})