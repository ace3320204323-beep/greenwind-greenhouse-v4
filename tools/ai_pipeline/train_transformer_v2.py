"""绿风润耕：单棚气候 Transformer 训练与评估脚本 v0.2。

设计目标：
1. 与 generate_data_v2.py 的字段契约一致；
2. 任何滑动窗口都不能跨 group_id 边界；
3. 默认按独立仿真组划分训练/验证/测试，避免相邻样本泄漏；
4. 归一化统计量只由训练组计算；
5. 在摄氏度原始尺度上，与持续法、线性趋势法公平对照；
6. truth_* 与 heater_efficiency_truth 只用于审计，禁止作为模型输入；
7. 预测未来温度相对当前温度的变化量，降低跨场景绝对温度偏移；
8. 保存 state_dict、元数据、训练日志和最终评估结果。

快速冒烟测试：
    python tools/ai_pipeline/train_transformer_v2.py ^
      --data tools/ai_pipeline/data/dataset_v2.csv ^
      --out tools/ai_pipeline/outputs_v2_smoke ^
      --epochs 2 --max-samples 4096

正式训练：
    python tools/ai_pipeline/train_transformer_v2.py ^
      --data tools/ai_pipeline/data/dataset_v2.csv ^
      --out tools/ai_pipeline/outputs_v2 ^
      --epochs 25
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


VERSION = "0.2.1-sim-residual"
TARGET = "t_center_c"

# 仅使用部署时能够获得的观测、时间和历史设备状态。
BASE_FEATURES = [
    "t_east_c",
    "t_center_c",
    "t_west_c",
    "rh_center_pct",
    "canopy_temp_c",
    "substrate_temp_c",
    "substrate_moisture_pct",
    "co2_ppm",
    "out_temp_c",
    "out_rh_pct",
    "out_solar_w_m2",
    "out_wind_m_s",
    "heater_cmd_pct",
    "vent_cmd_pct",
    "curtain_cmd_pct",
    "fan_cmd_pct",
    "heater_actual_pct",
    "vent_actual_pct",
    "curtain_actual_pct",
    "fan_actual_pct",
    "is_day",
]

QUALITY_COLUMNS = [
    "t_east_quality",
    "t_center_quality",
    "t_west_quality",
    "rh_center_quality",
    "out_temp_quality",
    "out_rh_quality",
    "out_solar_quality",
    "out_wind_quality",
]

FORBIDDEN_INPUT_PREFIXES = ("truth_",)
FORBIDDEN_INPUTS = {"heater_efficiency_truth"}


@dataclass(frozen=True)
class WindowRef:
    group_id: int
    start: int


@dataclass(frozen=True)
class SplitInfo:
    train_groups: list[int]
    val_groups: list[int]
    test_groups: list[int]


class ClimateTransformer(nn.Module):
    """轻量时间序列 Transformer，输出多个预测时长的中心棚温。"""

    def __init__(
        self,
        feature_dim: int,
        window: int,
        output_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x) + self.position_embedding[:, : x.size(1)]
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class GroupWindowDataset(Dataset):
    """按组保存数组并按需切片，保证窗口和标签不跨组。"""

    def __init__(
        self,
        arrays: dict[int, np.ndarray],
        target_index: int,
        groups: Sequence[int],
        window: int,
        horizons: Sequence[int],
        stride: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.target_index = target_index
        self.window = int(window)
        self.horizons = tuple(int(h) for h in horizons)
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.refs: list[WindowRef] = []
        max_horizon = max(self.horizons)
        for group_id in groups:
            length = len(arrays[group_id])
            final_start = length - self.window - max_horizon
            if final_start < 0:
                continue
            self.refs.extend(
                WindowRef(group_id, start)
                for start in range(0, final_start + 1, stride)
            )
        if not self.refs:
            raise ValueError("当前分组无法构造任何滑动窗口，请检查数据长度和窗口参数")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        ref = self.refs[index]
        group = self.arrays[ref.group_id]
        end = ref.start + self.window
        x_raw = group[ref.start:end]
        y_raw = np.asarray(
            [group[end - 1 + horizon, self.target_index] for horizon in self.horizons],
            dtype=np.float32,
        )
        x = (x_raw - self.x_mean) / self.x_std
        last_temp = np.float32(x_raw[-1, self.target_index])
        delta_raw = y_raw - last_temp
        y = (delta_raw - self.y_mean) / self.y_std
        # 以最后10分钟斜率构造线性趋势基线；不足10步时使用窗口首点。
        anchor = max(0, len(x_raw) - 11)
        elapsed = max(1, len(x_raw) - 1 - anchor)
        slope = np.float32((x_raw[-1, self.target_index] - x_raw[anchor, self.target_index]) / elapsed)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y),
            torch.from_numpy(y_raw),
            torch.tensor(last_temp, dtype=torch.float32),
            torch.tensor(slope, dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def quality_score(series: pd.Series) -> np.ndarray:
    """把质量标记转为数值特征，异常事实仍保留给模型。"""
    mapping = {
        "GOOD": 1.0,
        "SUSPECT": 0.5,
        "OUT_OF_RANGE": 0.0,
        "MISSING": 0.0,
        "STUCK": 0.0,
        "SPIKE": 0.0,
    }
    return series.astype(str).str.upper().map(mapping).fillna(0.0).to_numpy(np.float32)


def validate_frame(frame: pd.DataFrame) -> None:
    required = {"group_id", "sim_step", "hour", TARGET, *BASE_FEATURES, *QUALITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"数据缺少必要字段：{missing}")
    if frame.empty:
        raise ValueError("数据集为空")
    if frame[list(BASE_FEATURES)].isna().any().any():
        columns = frame[list(BASE_FEATURES)].columns[frame[list(BASE_FEATURES)].isna().any()].tolist()
        raise ValueError(f"模型输入存在缺失值：{columns}")
    if not frame["rh_center_pct"].between(0.0, 100.0).all():
        raise ValueError("rh_center_pct 超出0~100%物理范围")
    if (frame["out_solar_w_m2"] < 0.0).any():
        raise ValueError("out_solar_w_m2 存在负值")
    if (frame["out_wind_m_s"] < 0.0).any():
        raise ValueError("out_wind_m_s 存在负值")
    if frame.duplicated(["group_id", "sim_step"]).any():
        raise ValueError("存在重复的 group_id + sim_step")
    for group_id, group in frame.groupby("group_id", sort=False):
        steps = group["sim_step"].to_numpy()
        if len(steps) > 1 and not np.all(np.diff(steps) == 1):
            raise ValueError(f"group_id={group_id} 的 sim_step 不连续")


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.sort_values(["group_id", "sim_step"]).reset_index(drop=True).copy()
    radians = 2.0 * math.pi * frame["hour"].astype(float) / 24.0
    frame["hour_sin"] = np.sin(radians)
    frame["hour_cos"] = np.cos(radians)
    quality_features: list[str] = []
    for column in QUALITY_COLUMNS:
        feature = f"{column}_score"
        frame[feature] = quality_score(frame[column])
        quality_features.append(feature)
    features = [*BASE_FEATURES, "hour_sin", "hour_cos", *quality_features]
    forbidden = [
        feature
        for feature in features
        if feature in FORBIDDEN_INPUTS or feature.startswith(FORBIDDEN_INPUT_PREFIXES)
    ]
    if forbidden:
        raise RuntimeError(f"检测到禁止进入模型的审计字段：{forbidden}")
    return frame, features


def split_groups(group_ids: Sequence[int], seed: int) -> SplitInfo:
    ids = [int(group_id) for group_id in sorted(group_ids)]
    if len(ids) < 5:
        raise ValueError("至少需要5个独立组才能划分训练/验证/测试集")
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_count = max(1, round(len(ids) * 0.17))
    val_count = max(1, round(len(ids) * 0.17))
    train_count = len(ids) - val_count - test_count
    if train_count < 3:
        raise ValueError("训练组数量不足")
    return SplitInfo(
        train_groups=sorted(ids[:train_count]),
        val_groups=sorted(ids[train_count : train_count + val_count]),
        test_groups=sorted(ids[train_count + val_count :]),
    )


def build_arrays(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> dict[int, np.ndarray]:
    columns = list(features)
    arrays: dict[int, np.ndarray] = {}
    for group_id, group in frame.groupby("group_id", sort=True):
        arrays[int(group_id)] = group[columns].to_numpy(np.float32, copy=True)
    return arrays


def training_statistics(
    arrays: dict[int, np.ndarray],
    train_groups: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    train_rows = np.concatenate([arrays[group_id] for group_id in train_groups], axis=0)
    x_mean = train_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = train_rows.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    return x_mean, x_std


def residual_statistics(
    arrays: dict[int, np.ndarray],
    train_groups: Sequence[int],
    target_index: int,
    window: int,
    horizons: Sequence[int],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """只用训练组窗口计算各预测时长的温度变化量统计。"""
    max_horizon = max(horizons)
    residuals: list[list[np.ndarray]] = [[] for _ in horizons]
    for group_id in train_groups:
        values = arrays[group_id][:, target_index]
        starts = np.arange(0, len(values) - window - max_horizon + 1, stride)
        if len(starts) == 0:
            continue
        last_indices = starts + window - 1
        last_values = values[last_indices]
        for index, horizon in enumerate(horizons):
            future_values = values[last_indices + horizon]
            residuals[index].append((future_values - last_values).astype(np.float64))

    if any(not chunks for chunks in residuals):
        raise ValueError("训练组无法计算温度变化量统计")
    y_mean = np.asarray(
        [np.concatenate(chunks).mean() for chunks in residuals], dtype=np.float32
    )
    y_std = np.asarray(
        [np.concatenate(chunks).std() for chunks in residuals], dtype=np.float32
    )
    y_std[y_std < 1e-6] = 1.0
    return y_mean, y_std


def limited_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False)).tolist()
    return Subset(dataset, indices)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def normalized_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            total_loss += float(criterion(prediction, y).item())
            total_items += int(y.numel())
    return total_loss / max(1, total_items)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[list[dict], dict[str, torch.Tensor], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        items = 0
        for x, y, _, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            items += int(x.size(0))

        train_mse = running / max(1, items)
        val_mse = normalized_mse(model, val_loader, device)
        record = {"epoch": epoch, "train_mse_normalized": train_mse, "val_mse_normalized": val_mse}
        history.append(record)
        print(f"Epoch {epoch:02d}/{epochs}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}")

        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"验证集连续 {patience} 轮未改善，提前停止。")
                break

    if best_state is None:
        raise RuntimeError("训练未生成有效模型")
    return history, best_state, best_epoch


def metric_pair(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mae_c": round(float(np.mean(np.abs(error))), 4),
        "rmse_c": round(float(np.sqrt(np.mean(np.square(error)))), 4),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizons: Sequence[int],
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    persistences: list[np.ndarray] = []
    linear_predictions: list[np.ndarray] = []
    horizon_array = np.asarray(horizons, dtype=np.float32)[None, :]

    with torch.no_grad():
        for x, _, y_raw, last_temp, slope in loader:
            normalized = model(x.to(device)).cpu().numpy()
            delta_prediction = normalized * y_std[None, :] + y_mean[None, :]
            target = y_raw.numpy()
            persistence = np.repeat(last_temp.numpy()[:, None], len(horizons), axis=1)
            prediction = persistence + delta_prediction
            linear = last_temp.numpy()[:, None] + slope.numpy()[:, None] * horizon_array
            predictions.append(prediction)
            targets.append(target)
            persistences.append(persistence)
            linear_predictions.append(linear)

    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    persistence = np.concatenate(persistences, axis=0)
    linear = np.concatenate(linear_predictions, axis=0)
    result: dict[str, dict] = {}
    for index, horizon in enumerate(horizons):
        transformer_metrics = metric_pair(pred[:, index], true[:, index])
        persistence_metrics = metric_pair(persistence[:, index], true[:, index])
        linear_metrics = metric_pair(linear[:, index], true[:, index])
        baseline_mae = persistence_metrics["mae_c"]
        improvement = 100.0 * (baseline_mae - transformer_metrics["mae_c"]) / max(baseline_mae, 1e-9)
        result[f"horizon_{horizon}min"] = {
            "transformer": transformer_metrics,
            "persistence": persistence_metrics,
            "linear_trend": linear_metrics,
            "improvement_vs_persistence_pct": round(improvement, 2),
        }
    return {"samples": int(len(true)), "metrics": result}


def json_ready(values: Iterable[float]) -> list[float]:
    return [round(float(value), 8) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="训练单棚气候 Transformer v0.2")
    parser.add_argument("--data", default="tools/ai_pipeline/data/dataset_v2.csv")
    parser.add_argument("--out", default="tools/ai_pipeline/outputs_v2")
    parser.add_argument("--window", type=int, default=60, help="历史窗口，单位：分钟")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="每个数据集最多抽取多少窗口；0表示全部。仅建议冒烟测试使用。",
    )
    args = parser.parse_args()

    if args.window < 2 or min(args.horizons) < 1:
        raise SystemExit("window必须>=2，horizons必须为正整数")
    if args.d_model % args.nhead != 0:
        raise SystemExit("d_model 必须能被 nhead 整除")

    set_seed(args.seed)
    data_path = Path(args.data)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取数据：{data_path}")
    frame = pd.read_csv(data_path)
    validate_frame(frame)
    frame, features = prepare_frame(frame)
    split = split_groups(frame["group_id"].unique().tolist(), args.seed)
    arrays = build_arrays(frame, features)
    target_index = features.index(TARGET)
    x_mean, x_std = training_statistics(arrays, split.train_groups)
    y_mean, y_std = residual_statistics(
        arrays=arrays,
        train_groups=split.train_groups,
        target_index=target_index,
        window=args.window,
        horizons=args.horizons,
        stride=args.stride,
    )

    common = dict(
        arrays=arrays,
        target_index=target_index,
        window=args.window,
        horizons=args.horizons,
        stride=args.stride,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    train_full = GroupWindowDataset(groups=split.train_groups, **common)
    val_full = GroupWindowDataset(groups=split.val_groups, **common)
    test_full = GroupWindowDataset(groups=split.test_groups, **common)
    train_data = limited_dataset(train_full, args.max_samples, args.seed)
    val_data = limited_dataset(val_full, args.max_samples, args.seed + 1)
    test_data = limited_dataset(test_full, args.max_samples, args.seed + 2)

    print(f"训练组={split.train_groups}，验证组={split.val_groups}，测试组={split.test_groups}")
    print(f"窗口数：train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    print(f"输入特征数={len(features)}，输入窗口={args.window}分钟，预测={args.horizons}分钟")

    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    val_loader = make_loader(val_data, args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)
    device = torch.device("cpu")
    model = ClimateTransformer(
        feature_dim=len(features),
        window=args.window,
        output_dim=len(args.horizons),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    history, best_state, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    model.load_state_dict(best_state)
    evaluation = evaluate(model, test_loader, device, args.horizons, y_mean, y_std)

    model_path = output_dir / "transformer_v2.pt"
    meta_path = output_dir / "transformer_v2_meta.json"
    log_path = output_dir / "train_log_v2.json"
    evaluation_path = output_dir / "evaluation_v2.json"
    torch.save(best_state, model_path)

    metadata = {
        "model": "ClimateTransformer",
        "version": VERSION,
        "data_source": "virtual_greenhouse_simulation",
        "window_minutes": args.window,
        "horizons_minutes": args.horizons,
        "features": features,
        "target": TARGET,
        "target_representation": "future_temperature_minus_current_temperature",
        "x_mean": json_ready(x_mean),
        "x_std": json_ready(x_std),
        "y_mean": json_ready(y_mean),
        "y_std": json_ready(y_std),
        "architecture": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "split": {"method": "independent_group", **asdict(split)},
        "best_epoch": best_epoch,
        "test_metrics": evaluation,
        "forbidden_inputs": ["truth_*", "heater_efficiency_truth"],
        "note": "当前权重仅由虚拟大棚数据训练，用于管线联调与仿真验证；接入真实棚室后需重新训练或微调并独立评估。",
    }
    train_log = {
        "version": VERSION,
        "arguments": vars(args),
        "split": asdict(split),
        "full_window_counts": {
            "train": len(train_full),
            "validation": len(val_full),
            "test": len(test_full),
        },
        "used_window_counts": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
        },
        "best_epoch": best_epoch,
        "history": history,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(train_log, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n测试集结果：")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(f"\n模型：{model_path}")
    print(f"元数据：{meta_path}")
    print(f"训练日志：{log_path}")
    print(f"评估结果：{evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
