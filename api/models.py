from typing import Annotated

from fastapi import (
    APIRouter, Form,
    status, HTTPException, 
    Depends
)
from schemas.diamond import (
    Diamond,
    Cut, Color, Clarity
    )



router = APIRouter(
    prefix="/models",
    tags=["Diamond Model"]
)

@router.post("/predictions/prices", status_code=status.HTTP_200_OK)
async def predict_price(
    carat: Annotated[float, Form()],
    cut: Annotated[Cut, Form()],
    color: Annotated[Color, Form()],
    clarity: Annotated[Clarity, Form()],
    depth: Annotated[float, Form()],
    table: Annotated[float, Form()],
    x: Annotated[float, Form()],
    y: Annotated[float, Form()],
    z: Annotated[float, Form()]
    ):
    return {"price": 100}

    