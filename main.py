import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from api import models, diamonds
from utils.config import DATASET_ROOT

STATIC_PATH = os.path.join(DATASET_ROOT, "diamonds", "_md-images")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(diamonds.router.prefix + "/_md-images", StaticFiles(directory=STATIC_PATH), name="static")

app.include_router(models.router)
app.include_router(diamonds.router)


@app.get("/", response_class=HTMLResponse)
async def welcome():
    message = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diamond API</title>
</head>
<body>
    <h1>Welcome to the Diamonds API</h1>
    
    <ul>
        <li>available routes:</li>
        <ul>
            <li><a href="/models/predictions/prices">/models/predictions/prices</a></li>
            <li><a href="/diamonds/info">/diamonds/info</a></li>
        </ul>
        <li>available usage:</li>
        <ul>
            <li>GET: <a href="/diamonds/info">/diamonds/info</a></li>
            <p>Explanation of the input parameters based on the model features</p>
            <li>POST: <a href="/models/predictions/prices">/models/predictions/prices</a></li>
            <p>Model prediction execution based on the input parameters</p>
            <p>To run the model prediction, please go to the /docs route and use the model prediction form:</p>
            <p><a href="http://localhost:8000/docs#/Diamond%20Model/predict_price_models_predictions_prices_post">http://localhost:8000/docs</a></p>
            <p>And press the "Try it out" button.</p>
        </ul>
    </ul>
    
    <p>Thank you for letting me serve you.</p>

</body>
</html>
"""
    return message
