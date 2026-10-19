"""绿风润耕单棚智能闭环统一运行时 v1.0。

把现有五个已经验证的环节串成一条可重复运行、可追溯的链路：

    虚拟大棚 -> 分钟观测 -> Transformer -> 28维状态 -> SAC V3
    -> 上层策略 -> 约束滚动控制 -> 安全裁决 -> 设备反馈 -> 日志/前端

本文件不重新实现 Transformer、SAC 或温室物理模型，只负责统一编排。
真实大棚接入时应替换数据源/设备适配器，保留相同的字段契约与日志契约。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


AI_DIR = Path(__file__).resolve().parent
VALID_MODES = {"auto", "advisory", "manual", "predictive", "fixed", "human_review"}
STRATEGY_NAMES = (
    "temperature_offset_c",
    "comfort_weight",
    "energy_weight",
    "dehumidify_priority",
)
DEVICE_NAMES = ("heater", "vent", "curtain", "fan")


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_to_strategy(action: Sequence[float]) -> Dict[str, float]:
    """把 SAC 的 [-1, 1] 四维动作还原为可读上层策略。"""
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw.shape != (4,):
        raise ValueError("SAC V3 策略动作必须为4维")
    raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    return {
        "temperature_offset_c": float(raw[0] * 2.0),
        "comfort_weight": float((raw[1] + 1.0) * 0.5),
        "energy_weight": float((raw[2] + 1.0) * 0.5),
        "dehumidify_priority": float((raw[3] + 1.0) * 0.5),
    }


def strategy_to_normalized(strategy: Mapping[str, Any]) -> np.ndarray:
    offset = float(np.clip(_finite(strategy.get("temperature_offset_c")), -2.0, 2.0))
    comfort = float(np.clip(_finite(strategy.get("comfort_weight"), 0.8), 0.0, 1.0))
    energy = float(np.clip(_finite(strategy.get("energy_weight"), 0.5), 0.0, 1.0))
    dehumidify = float(
        np.clip(_finite(strategy.get("dehumidify_priority"), 0.3), 0.0, 1.0)
    )
    return np.asarray(
        [offset / 2.0, comfort * 2.0 - 1.0, energy * 2.0 - 1.0, dehumidify * 2.0 - 1.0],
        dtype=np.float32,
    )


@dataclass
class RuntimeConfig:
    source: str = "virtual_greenhouse"
    mode: str = "auto"
    seed: int = 42
    episode_hours: float = 24.0
    control_interval_minutes: int = 5
    warmup_minutes: int = 60
    minimum_hold_steps: int = 2
    max_control_steps: Optional[int] = 12
    scenario_path: Path = AI_DIR.parent / "virtual_greenhouse" / "scenarios" / "shandong_winter_strawberry.json"
    transformer_model_path: Path = AI_DIR / "outputs_v2_1_residual" / "transformer_v2.pt"
    transformer_meta_path: Path = AI_DIR / "outputs_v2_1_residual" / "transformer_v2_meta.json"
    sac_model_path: Path = AI_DIR / "sac_runs" / "formal_v3" / "sac_v3_final.zip"
    output_root: Path = AI_DIR / "integrated_runs"
    review_root: Path = AI_DIR.parent.parent / "var" / "ai_review"
    human_review_timeout_seconds: float = 30.0
    human_review_poll_seconds: float = 0.20
    print_progress: bool = True
    scenario_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "base_outdoor_temp_c": 5.0,
            "weather_scenes": ["clear_cold"],
            "fault_schedule": [],
        }
    )
    manual_strategy: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature_offset_c": 0.0,
            "comfort_weight": 0.8,
            "energy_weight": 0.5,
            "dehumidify_priority": 0.3,
        }
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimeConfig":
        config_path = Path(path).resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        return cls(
            source=str(data.get("source", "virtual_greenhouse")),
            mode=str(data.get("mode", "auto")).lower(),
            seed=int(data.get("seed", 42)),
            episode_hours=float(data.get("episode_hours", 24.0)),
            control_interval_minutes=int(data.get("control_interval_minutes", 5)),
            warmup_minutes=int(data.get("warmup_minutes", 60)),
            minimum_hold_steps=int(data.get("minimum_hold_steps", 2)),
            max_control_steps=(
                None
                if data.get("max_control_steps") is None
                else int(data["max_control_steps"])
            ),
            scenario_path=_resolve(base, data["scenario_path"]),
            transformer_model_path=_resolve(base, data["transformer_model_path"]),
            transformer_meta_path=_resolve(base, data["transformer_meta_path"]),
            sac_model_path=_resolve(base, data["sac_model_path"]),
            output_root=_resolve(base, data.get("output_root", "integrated_runs")),
            review_root=_resolve(base, data.get("review_root", "../../var/ai_review")),
            human_review_timeout_seconds=float(
                data.get("human_review_timeout_seconds", 30.0)
            ),
            human_review_poll_seconds=float(data.get("human_review_poll_seconds", 0.20)),
            print_progress=bool(data.get("print_progress", True)),
            scenario_options=dict(data.get("scenario_options", {})),
            manual_strategy=dict(data.get("manual_strategy", {})),
        )

    def validate(self, check_files: bool = True) -> None:
        if self.source != "virtual_greenhouse":
            raise ValueError(
                "当前正式运行时只接入 virtual_greenhouse；真实棚接入需实现现场适配器后再切换。"
            )
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode必须属于{sorted(VALID_MODES)}，当前为{self.mode!r}")
        if self.episode_hours <= 0:
            raise ValueError("episode_hours必须大于0")
        if self.control_interval_minutes < 1:
            raise ValueError("control_interval_minutes必须大于等于1")
        if self.warmup_minutes < 0:
            raise ValueError("warmup_minutes不能为负数")
        if self.max_control_steps is not None and self.max_control_steps < 1:
            raise ValueError("max_control_steps必须为空或大于等于1")
        if self.human_review_timeout_seconds <= 0:
            raise ValueError("human_review_timeout_seconds必须大于0")
        if self.human_review_poll_seconds <= 0:
            raise ValueError("human_review_poll_seconds必须大于0")
        strategy_to_normalized(self.manual_strategy)
        if check_files:
            required = {
                "scenario_path": self.scenario_path,
                "transformer_model_path": self.transformer_model_path,
                "transformer_meta_path": self.transformer_meta_path,
            }
            if self.mode in {"auto", "advisory", "human_review"}:
                required["sac_model_path"] = self.sac_model_path
            missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError("缺少运行文件：" + "; ".join(missing))

    def public_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


class TraceWriter:
    """固定字段、逐行刷新，保证运行中断后仍能保留已完成记录。"""

    TABLES: Dict[str, Sequence[str]] = {
        "state_trace.csv": (
            "control_step", "timestamp", "sim_hour", "temperature_c",
            "relative_humidity_pct", "target_low_c", "target_high_c",
            "temperature_compliant", "reward", "terminated", "truncated",
            "out_temp_c", "out_rh_pct", "out_solar_w_m2", "out_wind_m_s",
        ),
        "prediction_trace.csv": (
            "control_step", "timestamp", "transformer_status", "transformer_quality",
            "current_temperature_c", "pred_10min_c", "pred_30min_c",
        ),
        "strategy_trace.csv": (
            "control_step", "timestamp", "runtime_mode", "applied_policy",
            "recommendation_only", "temperature_offset_c", "comfort_weight",
            "energy_weight", "dehumidify_priority", "sac_temperature_offset_c",
            "sac_comfort_weight", "sac_energy_weight", "sac_dehumidify_priority",
            "planner_objective",
        ),
        "command_trace.csv": (
            "control_step", "timestamp",
            "heater_requested_pct", "vent_requested_pct", "curtain_requested_pct", "fan_requested_pct",
            "heater_safe_pct", "vent_safe_pct", "curtain_safe_pct", "fan_safe_pct",
            "heater_actual_pct", "vent_actual_pct", "curtain_actual_pct", "fan_actual_pct",
        ),
        "safety_trace.csv": (
            "control_step", "timestamp", "safety_override_count", "safety_overrides_json",
            "planner_constraint_count", "planner_constraints_json", "transformer_degraded",
        ),
        "human_review_trace.csv": (
            "control_step", "decision_id", "requested_at", "responded_at",
            "decision", "operator", "latency_seconds", "result_policy",
            "ai_strategy_json", "applied_strategy_json", "note",
        ),
    }

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._handles: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        for filename, fields in self.TABLES.items():
            handle = (run_dir / filename).open("w", encoding="utf-8-sig", newline="")
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            handle.flush()
            self._handles[filename] = handle
            self._writers[filename] = writer

    def row(self, filename: str, values: Mapping[str, Any]) -> None:
        self._writers[filename].writerow({key: _jsonable(value) for key, value in values.items()})
        self._handles[filename].flush()

    def event(self, event: Mapping[str, Any]) -> None:
        with (self.run_dir / "runtime_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")

    def latest(self, payload: Mapping[str, Any]) -> None:
        target = self.run_dir / "latest_state.json"
        temporary = self.run_dir / "latest_state.tmp"
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass


class HumanReviewBroker:
    """基于本地 JSON 文件的人机审核通道，不开放网络端口。"""

    VALID_DECISIONS = {"approve", "modify", "reject"}

    def __init__(self, root: Path, timeout_seconds: float, poll_seconds: float) -> None:
        self.root = Path(root)
        self.pending_file = self.root / "pending_decision.json"
        self.response_dir = self.root / "responses"
        self.archive_dir = self.root / "archive"
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _clear_pending(self, decision_id: str) -> None:
        try:
            current = json.loads(self.pending_file.read_text(encoding="utf-8"))
            if current.get("decision_id") == decision_id:
                self.pending_file.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def request(
        self,
        step: int,
        info: Mapping[str, Any],
        recommended_action: np.ndarray,
        fixed_action: np.ndarray,
        stop_event: threading.Event,
    ) -> tuple[np.ndarray, str, Dict[str, Any]]:
        requested_at = datetime.now(timezone.utc)
        decision_id = f"step-{step + 1:05d}-{uuid.uuid4().hex[:10]}"
        ai_strategy = normalized_to_strategy(recommended_action)
        low, high = info.get("target_band_c", (8.0, 10.0))
        pending = {
            "schema_version": "1.0",
            "decision_id": decision_id,
            "control_step": step + 1,
            "requested_at": requested_at.isoformat(),
            "deadline_at": datetime.fromtimestamp(
                requested_at.timestamp() + self.timeout_seconds, tz=timezone.utc
            ).isoformat(),
            "state": {
                "sim_hour": _finite(info.get("sim_hour")),
                "temperature_c": _finite(info.get("temperature_c")),
                "relative_humidity_pct": _finite(info.get("relative_humidity_pct")),
                "target_band_c": [_finite(low), _finite(high)],
            },
            "prediction": {
                "status": info.get("transformer_status", "UNKNOWN"),
                "quality": info.get("transformer_quality", "UNKNOWN"),
                "temperature_10min_c": _finite(info.get("pred_10min_c")),
                "temperature_30min_c": _finite(info.get("pred_30min_c")),
            },
            "ai_strategy": ai_strategy,
            "allowed_decisions": ["approve", "modify", "reject"],
            "timeout_fallback": "fixed_safe_strategy",
            "notice": "人工决定仍须经过约束控制器与安全裁决，不能直接驱动设备。",
        }
        self._write_atomic(self.pending_file, pending)
        response_path = self.response_dir / f"{decision_id}.json"
        deadline = time.monotonic() + self.timeout_seconds
        response: Dict[str, Any] = {}
        result_policy = "human_timeout_fixed"
        applied = fixed_action.copy()

        while time.monotonic() < deadline and not stop_event.is_set():
            if response_path.is_file():
                try:
                    candidate = json.loads(response_path.read_text(encoding="utf-8"))
                    if candidate.get("decision_id") != decision_id:
                        raise ValueError("decision_id不匹配")
                    decision = str(candidate.get("decision", "")).lower()
                    if decision not in self.VALID_DECISIONS:
                        raise ValueError("decision必须为approve/modify/reject")
                    response = candidate
                    if decision == "approve":
                        applied = recommended_action.copy()
                        result_policy = "human_approved_sac"
                    elif decision == "modify":
                        applied = strategy_to_normalized(candidate.get("strategy", {}))
                        result_policy = "human_modified_strategy"
                    else:
                        applied = fixed_action.copy()
                        result_policy = "human_rejected_fixed"
                    break
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    response = {
                        "decision": "invalid_response",
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                    time.sleep(self.poll_seconds)
                    continue
            stop_event.wait(self.poll_seconds)

        responded_at = datetime.now(timezone.utc)
        decision = str(response.get("decision", "timeout"))
        if stop_event.is_set() and not response:
            decision = "stopped"
            result_policy = "human_stop_fixed"
        review = {
            "decision_id": decision_id,
            "requested_at": requested_at.isoformat(),
            "responded_at": responded_at.isoformat(),
            "decision": decision,
            "operator": response.get("operator", "LOCAL_OPERATOR"),
            "latency_seconds": round(responded_at.timestamp() - requested_at.timestamp(), 3),
            "result_policy": result_policy,
            "ai_strategy": ai_strategy,
            "applied_strategy": normalized_to_strategy(applied),
            "note": response.get("note", ""),
        }
        self._write_atomic(self.archive_dir / f"{decision_id}.json", {**pending, "review": review})
        response_path.unlink(missing_ok=True)
        self._clear_pending(decision_id)
        return applied, result_policy, review


class IntegratedRuntime:
    """统一编排器；允许注入假环境/假策略以进行无模型单元测试。"""

    def __init__(
        self,
        config: RuntimeConfig,
        env_factory: Optional[Callable[[RuntimeConfig], Any]] = None,
        policy_loader: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        self.config = config
        self._env_factory = env_factory
        self._policy_loader = policy_loader
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._running = False
        self._env: Any = None
        self._sac_policy: Any = None
        self._review_broker: Optional[HumanReviewBroker] = None
        self._current_review: Optional[Dict[str, Any]] = None
        self.run_dir: Optional[Path] = None

    def request_stop(self) -> bool:
        """幂等停止请求：第一次返回True，重复请求返回False，不阻塞调用线程。"""
        with self._stop_lock:
            already = self._stop_event.is_set()
            self._stop_event.set()
            return not already

    def _make_env(self) -> Any:
        if self._env_factory is not None:
            return self._env_factory(self.config)
        from greenhouse_sac_env_v3 import GreenhouseSACV3Env

        return GreenhouseSACV3Env(
            scenario_path=self.config.scenario_path,
            model_path=self.config.transformer_model_path,
            meta_path=self.config.transformer_meta_path,
            episode_hours=self.config.episode_hours,
            control_interval_minutes=self.config.control_interval_minutes,
            warmup_minutes=self.config.warmup_minutes,
            use_transformer=True,
            minimum_hold_steps=self.config.minimum_hold_steps,
        )

    def _load_policy(self) -> Any:
        if self._policy_loader is not None:
            return self._policy_loader(self.config.sac_model_path)
        from stable_baselines3 import SAC

        return SAC.load(self.config.sac_model_path, device="cpu")

    @staticmethod
    def _sac_action(policy: Any, observation: np.ndarray) -> np.ndarray:
        predicted = policy.predict(observation, deterministic=True)
        action = predicted[0] if isinstance(predicted, tuple) else predicted
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape != (4,):
            raise ValueError(f"SAC模型输出应为4维，实际为{raw.shape}")
        return np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    @staticmethod
    def _fixed_action(info: Mapping[str, Any]) -> np.ndarray:
        rh = _finite(info.get("relative_humidity_pct"), 70.0)
        return strategy_to_normalized(
            {
                "temperature_offset_c": 0.0,
                "comfort_weight": 0.80,
                "energy_weight": 0.55,
                "dehumidify_priority": 0.85 if rh > 88.0 else 0.30,
            }
        )

    @staticmethod
    def _predictive_action(info: Mapping[str, Any]) -> np.ndarray:
        pred10 = _finite(info.get("pred_10min_c"), info.get("temperature_c", 15.0))
        pred30 = _finite(info.get("pred_30min_c"), info.get("temperature_c", 15.0))
        rh = _finite(info.get("relative_humidity_pct"), 70.0)
        band = info.get("target_band_c", (8.0, 10.0))
        low, high = _finite(band[0], 8.0), _finite(band[1], 10.0)
        if pred30 < low:
            offset = min(2.0, 0.45 * (low - pred30))
        elif pred30 > high:
            offset = max(-2.0, -0.45 * (pred30 - high))
        else:
            offset = 0.0
        projected_error = max(0.0, low - pred30, pred30 - high)
        short_error = max(0.0, low - pred10, pred10 - high)
        return strategy_to_normalized(
            {
                "temperature_offset_c": offset,
                "comfort_weight": min(1.0, 0.60 + 0.12 * projected_error + 0.05 * short_error),
                "energy_weight": max(0.25, 0.65 - 0.10 * projected_error),
                "dehumidify_priority": min(1.0, max(0.20, (rh - 75.0) / 20.0)),
            }
        )

    def _select_action(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> tuple[np.ndarray, Optional[np.ndarray], str]:
        mode = self.config.mode
        sac_action: Optional[np.ndarray] = None
        if mode in {"auto", "advisory", "human_review"}:
            sac_action = self._sac_action(self._sac_policy, observation)
        if mode in {"auto", "human_review"}:
            return sac_action.copy(), sac_action, "sac_v3"
        if mode == "advisory":
            return self._predictive_action(info), sac_action, "predictive_strategy"
        if mode == "manual":
            return strategy_to_normalized(self.config.manual_strategy), None, "manual_strategy"
        if mode == "predictive":
            return self._predictive_action(info), None, "predictive_strategy"
        return self._fixed_action(info), None, "fixed_strategy"

    def validate(self) -> Dict[str, Any]:
        injected = self._env_factory is not None
        self.config.validate(check_files=not injected)
        artifacts: Dict[str, Any] = {}
        if not injected:
            for name, path in {
                "scenario": self.config.scenario_path,
                "transformer_model": self.config.transformer_model_path,
                "transformer_meta": self.config.transformer_meta_path,
                "sac_model": self.config.sac_model_path,
            }.items():
                if path.is_file():
                    artifacts[name] = {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
        return {
            "status": "OK",
            "architecture": "virtual_greenhouse_transformer_sac_v3_constrained_control",
            "observation_dimension": 28,
            "strategy_dimension": 4,
            "device_dimension": 4,
            "config": self.config.public_dict(),
            "artifacts": artifacts,
        }

    def _new_run_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = self.config.output_root / f"run_{stamp}_seed{self.config.seed}"
        candidate = base
        index = 1
        while candidate.exists():
            candidate = Path(f"{base}_{index}")
            index += 1
        return candidate

    def _write_step(
        self,
        traces: TraceWriter,
        step: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        applied_action: np.ndarray,
        sac_action: Optional[np.ndarray],
        applied_policy: str,
    ) -> Dict[str, Any]:
        record = dict(getattr(self._env, "_latest_record", {}) or {})
        timestamp = str(record.get("timestamp", f"sim_hour_{_finite(info.get('sim_hour')):.4f}"))
        low, high = map(float, info.get("target_band_c", (0.0, 0.0)))
        temperature = _finite(info.get("temperature_c"))
        strategy = dict(info.get("strategy_action", normalized_to_strategy(applied_action)))
        sac_strategy = normalized_to_strategy(sac_action) if sac_action is not None else {}
        requested = dict(info.get("requested_targets_pct", {}))
        safe = dict(info.get("safe_targets_pct", {}))
        actual = dict(info.get("actuals_pct", {}))
        overrides = tuple(info.get("safety_overrides", ()))
        constraints = tuple(info.get("planner_constraint_events", ()))

        traces.row(
            "state_trace.csv",
            {
                "control_step": step,
                "timestamp": timestamp,
                "sim_hour": _finite(info.get("sim_hour")),
                "temperature_c": temperature,
                "relative_humidity_pct": _finite(info.get("relative_humidity_pct")),
                "target_low_c": low,
                "target_high_c": high,
                "temperature_compliant": int(low <= temperature <= high),
                "reward": float(reward),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "out_temp_c": _finite(record.get("out_temp_c")),
                "out_rh_pct": _finite(record.get("out_rh_pct")),
                "out_solar_w_m2": _finite(record.get("out_solar_w_m2")),
                "out_wind_m_s": _finite(record.get("out_wind_m_s")),
            },
        )
        traces.row(
            "prediction_trace.csv",
            {
                "control_step": step,
                "timestamp": timestamp,
                "transformer_status": info.get("transformer_status", "UNKNOWN"),
                "transformer_quality": info.get("transformer_quality", "UNKNOWN"),
                "current_temperature_c": temperature,
                "pred_10min_c": _finite(info.get("pred_10min_c"), temperature),
                "pred_30min_c": _finite(info.get("pred_30min_c"), temperature),
            },
        )
        traces.row(
            "strategy_trace.csv",
            {
                "control_step": step,
                "timestamp": timestamp,
                "runtime_mode": self.config.mode,
                "applied_policy": applied_policy,
                "recommendation_only": int(self.config.mode == "advisory"),
                **strategy,
                **{f"sac_{key}": value for key, value in sac_strategy.items()},
                "planner_objective": _finite(info.get("planner_objective")),
            },
        )
        command_row: Dict[str, Any] = {"control_step": step, "timestamp": timestamp}
        for name in DEVICE_NAMES:
            command_row[f"{name}_requested_pct"] = _finite(requested.get(name))
            command_row[f"{name}_safe_pct"] = _finite(safe.get(name))
            command_row[f"{name}_actual_pct"] = _finite(actual.get(name))
        traces.row("command_trace.csv", command_row)
        traces.row(
            "safety_trace.csv",
            {
                "control_step": step,
                "timestamp": timestamp,
                "safety_override_count": len(overrides),
                "safety_overrides_json": json.dumps(overrides, ensure_ascii=False),
                "planner_constraint_count": len(constraints),
                "planner_constraints_json": json.dumps(constraints, ensure_ascii=False),
                "transformer_degraded": int(info.get("transformer_quality") != "GOOD"),
            },
        )
        if self._current_review is not None:
            traces.row(
                "human_review_trace.csv",
                {
                    "control_step": step,
                    **self._current_review,
                    "ai_strategy_json": json.dumps(
                        self._current_review.get("ai_strategy", {}), ensure_ascii=False
                    ),
                    "applied_strategy_json": json.dumps(
                        self._current_review.get("applied_strategy", {}), ensure_ascii=False
                    ),
                },
            )
        payload = {
            "runtime": {
                "status": "RUNNING",
                "mode": self.config.mode,
                "source": self.config.source,
                "control_step": step,
            },
            "state": {
                "timestamp": timestamp,
                "sim_hour": _finite(info.get("sim_hour")),
                "temperature_c": temperature,
                "relative_humidity_pct": _finite(info.get("relative_humidity_pct")),
                "target_band_c": [low, high],
                "raw_record": record,
            },
            "prediction": {
                "status": info.get("transformer_status", "UNKNOWN"),
                "quality": info.get("transformer_quality", "UNKNOWN"),
                "temperature_10min_c": _finite(info.get("pred_10min_c"), temperature),
                "temperature_30min_c": _finite(info.get("pred_30min_c"), temperature),
            },
            "strategy": {
                "applied_policy": applied_policy,
                "applied": strategy,
                "sac_recommendation": sac_strategy or None,
                "recommendation_only": self.config.mode == "advisory",
            },
            "control": {"requested_pct": requested, "safe_pct": safe, "actual_pct": actual},
            "safety": {
                "overrides": overrides,
                "planner_constraints": constraints,
            },
            "human_review": self._current_review,
            "reward": float(reward),
        }
        traces.latest(payload)
        return payload

    def run(self) -> Dict[str, Any]:
        if self._running:
            raise RuntimeError("运行时已在执行，不能重复启动")
        self._running = True
        self._stop_event.clear()
        validation = self.validate()
        self.run_dir = self._new_run_dir()
        traces = TraceWriter(self.run_dir)
        (self.run_dir / "runtime_manifest.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        (self.config.output_root / "latest_run.json").write_text(
            json.dumps({"run_dir": str(self.run_dir)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        started = time.perf_counter()
        step = 0
        rewards: list[float] = []
        temp_errors: list[float] = []
        compliant: list[int] = []
        override_steps = 0
        planner_steps = 0
        ended_reason = "unknown"
        error: Optional[str] = None
        try:
            self._env = self._make_env()
            if self.config.mode in {"auto", "advisory", "human_review"}:
                self._sac_policy = self._load_policy()
            if self.config.mode == "human_review":
                self._review_broker = HumanReviewBroker(
                    self.config.review_root,
                    self.config.human_review_timeout_seconds,
                    self.config.human_review_poll_seconds,
                )
            observation, info = self._env.reset(
                seed=self.config.seed,
                options={"episode_seed": self.config.seed, **self.config.scenario_options},
            )
            traces.event(
                {
                    "event": "runtime_started",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "mode": self.config.mode,
                    "source": self.config.source,
                }
            )
            while True:
                if self._stop_event.is_set():
                    ended_reason = "stopped_by_user"
                    break
                if self.config.max_control_steps is not None and step >= self.config.max_control_steps:
                    ended_reason = "max_control_steps"
                    break

                applied_action, sac_action, applied_policy = self._select_action(observation, info)
                self._current_review = None
                if self.config.mode == "human_review":
                    fixed_action = self._fixed_action(info)
                    applied_action, applied_policy, self._current_review = (
                        self._review_broker.request(
                            step=step,
                            info=info,
                            recommended_action=applied_action,
                            fixed_action=fixed_action,
                            stop_event=self._stop_event,
                        )
                    )
                observation, reward, terminated, truncated, info = self._env.step(applied_action)
                step += 1
                payload = self._write_step(
                    traces, step, float(reward), bool(terminated), bool(truncated), info,
                    applied_action, sac_action, applied_policy,
                )
                rewards.append(float(reward))
                temperature = _finite(info.get("temperature_c"))
                low, high = map(float, info.get("target_band_c", (0.0, 0.0)))
                error_now = max(0.0, low - temperature, temperature - high)
                temp_errors.append(error_now)
                compliant.append(int(error_now == 0.0))
                override_steps += int(bool(info.get("safety_overrides", ())))
                planner_steps += int(bool(info.get("planner_constraint_events", ())))
                if self.config.print_progress:
                    progress = {
                        "step": step,
                        "sim_hour": round(_finite(info.get("sim_hour")), 3),
                        "temperature_c": round(temperature, 3),
                        "pred_30min_c": round(_finite(info.get("pred_30min_c"), temperature), 3),
                        "policy": applied_policy,
                        "safe_pct": payload["control"]["safe_pct"],
                    }
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
                if terminated:
                    ended_reason = "terminated_invalid_or_extreme_state"
                    break
                if truncated:
                    ended_reason = "episode_completed"
                    break
        except KeyboardInterrupt:
            self.request_stop()
            ended_reason = "keyboard_interrupt"
        except Exception as exc:
            ended_reason = "runtime_error"
            error = f"{type(exc).__name__}: {exc}"
            traces.event(
                {
                    "event": "runtime_error",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "error": error,
                }
            )
            raise
        finally:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception as close_exc:
                    traces.event(
                        {
                            "event": "close_error",
                            "at": datetime.now(timezone.utc).isoformat(),
                            "error": f"{type(close_exc).__name__}: {close_exc}",
                        }
                    )
            elapsed = time.perf_counter() - started
            divisor = max(1, step)
            summary = {
                "status": "ERROR" if error else "COMPLETED",
                "ended_reason": ended_reason,
                "source": self.config.source,
                "mode": self.config.mode,
                "seed": self.config.seed,
                "steps": step,
                "simulated_control_minutes": step * self.config.control_interval_minutes,
                "wall_time_seconds": round(elapsed, 3),
                "reward_total": round(sum(rewards), 6),
                "reward_mean": round(sum(rewards) / divisor, 6),
                "temperature_error_mean_c": round(sum(temp_errors) / divisor, 6),
                "temperature_compliance_rate": round(sum(compliant) / divisor, 6),
                "safety_override_rate": round(override_steps / divisor, 6),
                "planner_constraint_rate": round(planner_steps / divisor, 6),
                "error": error,
                "run_dir": str(self.run_dir),
            }
            (self.run_dir / "runtime_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            traces.event(
                {
                    "event": "runtime_stopped",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                }
            )
            traces.close()
            self._running = False
        return summary
