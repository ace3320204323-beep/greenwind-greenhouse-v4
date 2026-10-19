"""绿风润耕 Transformer 实时推理接口。

把每分钟的一条传感器/设备记录送入 TransformerPredictor.update()。
缓存满60分钟后，返回未来10分钟、30分钟中心棚温预测。
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from train_transformer_v2 import ClimateTransformer


QUALITY_SCORE = {
    "GOOD": 1.0,
    "SUSPECT": 0.5,
    "OUT_OF_RANGE": 0.0,
    "MISSING": 0.0,
    "STUCK": 0.0,
    "SPIKE": 0.0,
}


class TransformerPredictor:
    """带60分钟滚动缓存的轻量Transformer推理器。"""

    def __init__(self, model_path: str | Path, meta_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.features = list(self.meta["features"])
        self.window = int(self.meta["window_minutes"])
        self.horizons = [int(value) for value in self.meta["horizons_minutes"]]
        self.version = str(self.meta["version"])
        self.target_representation = self.meta.get("target_representation", "absolute_temperature")
        self.x_mean = np.asarray(self.meta["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(self.meta["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(self.meta["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(self.meta["y_std"], dtype=np.float32)
        if self.y_mean.ndim == 0:
            self.y_mean = np.repeat(self.y_mean, len(self.horizons))
        if self.y_std.ndim == 0:
            self.y_std = np.repeat(self.y_std, len(self.horizons))

        architecture = self.meta["architecture"]
        self.model = ClimateTransformer(
            feature_dim=len(self.features),
            window=self.window,
            output_dim=len(self.horizons),
            d_model=int(architecture["d_model"]),
            nhead=int(architecture["nhead"]),
            num_layers=int(architecture["num_layers"]),
            dropout=float(architecture["dropout"]),
        )
        try:
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self.last_timestamp: str | None = None

    def reset(self) -> None:
        self.buffer.clear()
        self.last_timestamp = None

    @staticmethod
    def _number(record: Mapping[str, Any], name: str) -> float:
        value = record.get(name)
        if value is None:
            raise ValueError(f"缺少输入字段：{name}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"字段{name}不是有限数值：{value}")
        return number

    def _feature_value(self, record: Mapping[str, Any], feature: str) -> float:
        if feature == "hour_sin":
            return math.sin(2.0 * math.pi * self._number(record, "hour") / 24.0)
        if feature == "hour_cos":
            return math.cos(2.0 * math.pi * self._number(record, "hour") / 24.0)
        if feature.endswith("_quality_score"):
            source = feature[: -len("_score")]
            quality = str(record.get(source, "MISSING")).upper()
            return QUALITY_SCORE.get(quality, 0.0)
        return self._number(record, feature)

    def _vector(self, record: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            [self._feature_value(record, feature) for feature in self.features],
            dtype=np.float32,
        )

    def update(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """加入一条分钟记录并返回推理结果；缓存未满时返回WARMING_UP。"""
        timestamp = str(record.get("timestamp", ""))
        current_temp = self._number(record, "t_center_c")
        try:
            vector = self._vector(record)
        except (TypeError, ValueError) as exc:
            return {
                "status": "INVALID_INPUT",
                "timestamp": timestamp,
                "current_temperature_c": current_temp,
                "model_version": self.version,
                "reason": str(exc),
            }

        self.buffer.append(vector)
        self.last_timestamp = timestamp
        if len(self.buffer) < self.window:
            return {
                "status": "WARMING_UP",
                "timestamp": timestamp,
                "current_temperature_c": current_temp,
                "buffer_rows": len(self.buffer),
                "required_rows": self.window,
                "model_version": self.version,
            }

        quality_indices = [
            index for index, name in enumerate(self.features) if name.endswith("_quality_score")
        ]
        latest = self.buffer[-1]
        input_quality = (
            "GOOD"
            if not quality_indices or all(latest[index] >= 1.0 for index in quality_indices)
            else "DEGRADED"
        )

        raw = np.stack(self.buffer, axis=0)
        normalized = (raw - self.x_mean) / self.x_std
        tensor = torch.from_numpy(normalized[None, :, :].astype(np.float32))
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor).cpu().numpy()[0]
        inference_ms = (time.perf_counter() - started) * 1000.0

        decoded = output * self.y_std + self.y_mean
        if self.target_representation == "future_temperature_minus_current_temperature":
            predicted = current_temp + decoded
        else:
            predicted = decoded

        predictions = {
            f"{horizon}min": round(float(value), 4)
            for horizon, value in zip(self.horizons, predicted)
        }
        return {
            "status": "READY",
            "timestamp": timestamp,
            "current_temperature_c": round(current_temp, 4),
            "predictions_c": predictions,
            "horizons_minutes": self.horizons,
            "model_version": self.version,
            "input_quality": input_quality,
            "inference_ms": round(inference_ms, 4),
        }

