from typing import Annotated

from pydantic import ValidationError
from fastapi import (
    APIRouter, Form,
    status, HTTPException
)

from ml_injection_pipeline import prediction_pipeline
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

    form = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }

    try:
        Diamond(**form)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors()
        )

    try:
        price = prediction_pipeline(form)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    return {"price": float(price)}
