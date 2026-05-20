from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes.predict import router as predict_router
from app.services.inference import get_inference_service


APP_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("transistor-defect-inspection")

app = FastAPI(title="Transistor Defect Inspection")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.include_router(predict_router)


@app.on_event("startup")
async def load_model_on_startup() -> None:
    logger.info("Loading model weights. First startup can take a while...")
    get_inference_service()
    logger.info("Model weights loaded. App is ready for predictions.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
