from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from utils.config import DATASET_INFO_PATH


router = APIRouter(
    prefix="/diamonds",
    tags=["Diamond Information"]
)


@router.get("/info", response_class=HTMLResponse)
async def get_info():
    HTML_PATH = DATASET_INFO_PATH.replace('.md', '.html')
    with open(HTML_PATH, 'r') as f:
        html = f.read()
    return html
