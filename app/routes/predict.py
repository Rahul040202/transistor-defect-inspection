from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.inference import get_inference_service


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024


@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Upload a PNG, JPG, WEBP, or BMP image.",
            },
            status_code=400,
        )

    image_bytes = await file.read()
    if not image_bytes:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "The uploaded file is empty."},
            status_code=400,
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Image is too large. Please upload a file under 8 MB.",
            },
            status_code=400,
        )

    suffix = Path(file.filename or "").suffix or ".png"

    try:
        result = get_inference_service().predict_bytes(image_bytes, suffix=suffix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "filename": file.filename,
        },
    )
