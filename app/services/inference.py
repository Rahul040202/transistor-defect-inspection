from __future__ import annotations

import base64
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from pydantic import BaseModel, ConfigDict, Field


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = APP_DIR / "weights" / "patchcore_model.pt"


class VisualizationImages(BaseModel):
    """Base64 PNG images ready to use as <img src="..."> values."""

    original: str
    heatmap: str
    mask: str
    overlay: str


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: str
    is_anomaly: bool
    score: float = Field(ge=0.0)
    visualizations: VisualizationImages


class InferenceService:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.engine = Engine(
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )

    def predict_path(self, image_path: str | Path) -> PredictionResult:
        pred = self.engine.predict(model=self.model, data_path=str(image_path))[0]

        image = self._prediction_image(pred)
        anomaly_map = self._prediction_map(pred.anomaly_map)
        mask = self._prediction_map(pred.pred_mask) > 0

        anomaly_map = self._resize_float_map(anomaly_map, image.shape[:2])
        mask = self._resize_bool_mask(mask, image.shape[:2])

        is_anomaly = bool(self._tensor_scalar(pred.pred_label))
        score = float(self._tensor_scalar(pred.pred_score))
        label = "Anomaly" if is_anomaly else "Good"

        return PredictionResult(
            label=label,
            is_anomaly=is_anomaly,
            score=score,
            visualizations=VisualizationImages(
                original=self._encode_rgb_png(image),
                heatmap=self._encode_rgb_png(self._make_heatmap(image, anomaly_map)),
                mask=self._encode_rgb_png(self._make_mask(mask)),
                overlay=self._encode_rgb_png(self._make_overlay(image, mask)),
            ),
        )

    def predict_bytes(self, image_bytes: bytes, suffix: str = ".png") -> PredictionResult:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as image_file:
            image_file.write(image_bytes)
            image_file.flush()
            return self.predict_path(image_file.name)

    def _load_model(self) -> Patchcore:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")

        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )

        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(self.model_path, map_location=self.device)

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _prediction_image(pred: Any) -> np.ndarray:
        image = pred.image
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        return InferenceService._normalize_float_image(image[..., :3])

    @staticmethod
    def _prediction_map(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        return np.squeeze(value).astype(np.float32)

    @staticmethod
    def _tensor_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().flatten()[0].item()
        elif isinstance(value, np.ndarray):
            value = value.flatten()[0].item()
        elif isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)

    @staticmethod
    def _normalize_float_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        return np.clip(image, 0.0, 1.0)

    @staticmethod
    def _resize_float_map(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        value = value.astype(np.float32)
        if value.shape != shape:
            value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        value_min = float(np.min(value))
        value_max = float(np.max(value))
        if value_max > value_min:
            value = (value - value_min) / (value_max - value_min)
        return np.clip(value, 0.0, 1.0)

    @staticmethod
    def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        mask = mask.astype(np.uint8)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    @staticmethod
    def _make_heatmap(image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return cv2.addWeighted(image.astype(np.float32), 0.55, heatmap, 0.45, 0)

    @staticmethod
    def _make_mask(mask: np.ndarray) -> np.ndarray:
        mask_image = np.zeros((*mask.shape, 3), dtype=np.float32)
        mask_image[mask] = [1.0, 1.0, 1.0]
        return mask_image

    @staticmethod
    def _make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        overlay[mask] = [1.0, 0.0, 0.0]
        return cv2.addWeighted(image.astype(np.float32), 0.7, overlay.astype(np.float32), 0.3, 0)

    @staticmethod
    def _encode_rgb_png(image: np.ndarray) -> str:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Could not encode visualization image.")
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()
