"""绿风润耕 V4 交互运行层。

把虚拟大棚、Transformer、SAC V3、约束控制、安全裁决和前端统一到
一个本地文件通信闭环。前端只读 state.json，只向 commands 目录写入
JSON 命令；真实大棚接入时替换环境适配器，保留同一状态与命令契约。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from generate_data_v2 import commands_from_targets, feedback_actuals
from greenhouse_sac_env_v3 import GreenhouseSACV3Env
from integrated_runtime_v1 import normalized_to_strategy, strategy_to_normalized
from strategy_to_control_v3 import StrategyToControlV3


DEVICE_NAMES = ("heater", "vent", "curtain", "fan")
VALID_MODES = {"manual", "assist", "auto"}


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


@dataclass
class UIConfig:
    greenhouse_name: str
    crop: str
    source: str
    default_mode: str
    wall_interval_seconds: float
    assist_timeout_seconds: float
    episode_hours: float
    control_interval_minutes: int
    warmup_minutes: int
    minimum_hold_steps: int
    seed: int
    max_control_steps: int
    scenario_path: Path
    transformer_model_path: Path
    transformer_meta_path: Path
    sac_model_path: Path
    state_root: Path
    base_outdoor_temp_c: float
    weather_scenes: list[str]

    @classmethod
    def load(cls, path: Path) -> "UIConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        root = PROJECT_ROOT

        def resolved(key: str) -> Path:
            value = Path(data[key])
            return value if value.is_absolute() else (root / value).resolve()

        mode = str(data.get("default_mode", "assist")).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"default_mode必须属于{sorted(VALID_MODES)}")
        return cls(
            greenhouse_name=str(data.get("greenhouse_name", "山东冬季草莓大棚")),
            crop=str(data.get("crop", "草莓采果期")),
            source=str(data.get("source", "virtual_greenhouse")),
            default_mode=mode,
            wall_interval_seconds=max(0.1, float(data.get("wall_interval_seconds", 1.0))),
            assist_timeout_seconds=max(5.0, float(data.get("assist_timeout_seconds", 60.0))),
            episode_hours=float(data.get("episode_hours", 24.0)),
            control_interval_minutes=int(data.get("control_interval_minutes", 5)),
            warmup_minutes=int(data.get("warmup_minutes", 60)),
            minimum_hold_steps=int(data.get("minimum_hold_steps", 2)),
            seed=int(data.get("seed", 42)),
            max_control_steps=int(data.get("max_control_steps", 288)),
            scenario_path=resolved("scenario_path"),
            transformer_model_path=resolved("transformer_model_path"),
            transformer_meta_path=resolved("transformer_meta_path"),
            sac_model_path=resolved("sac_model_path"),
            state_root=resolved("state_root"),
            base_outdoor_temp_c=float(data.get("base_outdoor_temp_c", 5.0)),
            weather_scenes=list(data.get("weather_scenes", ["clear_cold"])),
        )

    def validate(self) -> None:
        required = (
            self.scenario_path,
            self.transformer_model_path,
            self.transformer_meta_path,
            self.sac_model_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("缺少运行文件：" + "; ".join(missing))
        if self.source != "virtual_greenhouse":
            raise ValueError("V4演示版当前只允许virtual_greenhouse数据源")


class InteractiveGreenhouseEnv(GreenhouseSACV3Env):
    """在 V3 AI 环境上增加人工设备目标入口，仍复用同一安全裁决。"""

    def step_manual_targets(self, targets: Mapping[str, Any]):
        if self._vg is None:
            raise RuntimeError("请先调用reset")
        requested = {
            name: float(np.clip(finite(targets.get(name), 0.0), 0.0, 100.0))
            for name in DEVICE_NAMES
        }
        safe, overrides = self._safety_filter(requested)
        safe_movement = float(
            np.mean(
                [abs(safe[name] - self._previous_targets[name]) / 100.0 for name in DEVICE_NAMES]
            )
        )
        raw_movement = float(
            np.mean(
                [abs(requested[name] - self._previous_targets[name]) / 100.0 for name in DEVICE_NAMES]
            )
        )
        overlap = min(max(0.0, requested["heater"] - 10.0), max(0.0, requested["vent"] - 10.0)) / 90.0
        energy_samples = []
        simulator_done = False
        for _ in range(self.control_interval_minutes):
            result = self._vg.step(commands_from_targets(safe))
            self._actuals = feedback_actuals(result.feedback, self._actuals)
            self._latest_batch = result.observation
            self._latest_record, self._latest_prediction = self._update_record_and_prediction(
                self._latest_batch, safe
            )
            energy_samples.append(
                self._actual("heater") / 100.0
                + 0.10 * self._actual("fan") / 100.0
                + 0.05 * self._actual("vent") / 100.0
                + 0.02 * self._actual("curtain") / 100.0
            )
            simulator_done = bool(result.done)
            if simulator_done:
                break

        self._decision_step += 1
        self._previous_targets = dict(safe)
        self._latest_plan = None
        self._strategy_movement_norm = 0.0
        reward, components = self._calculate_reward(
            safe_movement=safe_movement,
            raw_request_movement=raw_movement,
            energy_norm=float(np.mean(energy_samples)) if energy_samples else 0.0,
            unsafe_request=float(overlap > 0.0),
            conflict_intensity=float(overlap),
        )
        observation = self._make_observation()
        temperature = finite(self._latest_record.get("t_center_c"), 15.0)
        terminated = bool(not np.all(np.isfinite(observation)) or temperature < -20.0 or temperature > 60.0)
        truncated = bool(simulator_done or self._decision_step >= self._max_decision_steps)
        info = self._make_info(requested, safe, overrides, components)
        self._last_info = info
        return observation, reward, terminated, truncated, info


class InteractiveRuntime:
    PRESETS: Dict[str, Dict[str, float]] = {
        "warm": {"heater": 65.0, "vent": 0.0, "curtain": 100.0, "fan": 0.0},
        "cool": {"heater": 0.0, "vent": 75.0, "curtain": 80.0, "fan": 55.0},
        "dehumidify": {"heater": 20.0, "vent": 45.0, "curtain": 30.0, "fan": 65.0},
        "stop": {"heater": 0.0, "vent": 0.0, "curtain": 0.0, "fan": 0.0},
    }

    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self.config.validate()
        self.root = config.state_root
        self.state_file = self.root / "state.json"
        self.command_dir = self.root / "commands"
        self.archive_dir = self.root / "command_archive"
        self.stop_file = self.root / "stop.request"
        self.log_file = self.root / "interaction_trace.csv"
        self.root.mkdir(parents=True, exist_ok=True)
        self.command_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)

        self.mode = config.default_mode
        self.paused = False
        self.emergency_stop = False
        self.manual_targets = {name: 0.0 for name in DEVICE_NAMES}
        self.last_command: Dict[str, Any] = {}
        self.pending: Optional[Dict[str, Any]] = None
        self.pending_response: Optional[Dict[str, Any]] = None
        # 市电场景属于外部能源事件，不应被“AI建议等待人工确认”阻塞。
        # 收到按钮命令后，用上一周期的安全设备目标推进一个控制周期，
        # 让市电/储能功率和SOC立即形成新的闭环状态。
        self.grid_event_pending = False
        self.trend: deque[Dict[str, float]] = deque(maxlen=180)
        self.total_safety_overrides = 0
        self.total_manual_commands = 0
        self.total_ai_steps = 0
        self._last_payload: Dict[str, Any] = {}

        self.env = InteractiveGreenhouseEnv(
            scenario_path=config.scenario_path,
            model_path=config.transformer_model_path,
            meta_path=config.transformer_meta_path,
            episode_hours=config.episode_hours,
            control_interval_minutes=config.control_interval_minutes,
            warmup_minutes=config.warmup_minutes,
            use_transformer=True,
            minimum_hold_steps=config.minimum_hold_steps,
        )
        from stable_baselines3 import SAC

        self.policy = SAC.load(config.sac_model_path, device="cpu")
        self.observation, self.info = self.env.reset(
            seed=config.seed,
            options={
                "episode_seed": config.seed,
                "base_outdoor_temp_c": config.base_outdoor_temp_c,
                "weather_scenes": config.weather_scenes,
                "fault_schedule": [],
            },
        )
        self.manual_targets = dict(self.info.get("safe_targets_pct", self.manual_targets))
        self._ensure_log()
        self.publish(status="READY", control_source="initialization")

    def _ensure_log(self) -> None:
        if self.log_file.exists():
            return
        with self.log_file.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "timestamp", "control_step", "mode", "temperature_c", "humidity_pct",
                    "pred_10min_c", "pred_30min_c", "heater_safe", "vent_safe",
                    "curtain_safe", "fan_safe", "battery_soc_pct", "battery_power_kw",
                    "renewable_power_kw", "grid_power_kw", "load_power_kw", "grid_status",
                    "unserved_power_kw", "safety_overrides", "command_type",
                ]
            )

    def _log_step(self, command_type: str) -> None:
        safe = self.info.get("safe_targets_pct", {})
        energy = self.info.get("energy", {})
        with self.log_file.open("a", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    self.info.get("decision_step", 0), self.mode,
                    self.info.get("temperature_c", 0), self.info.get("relative_humidity_pct", 0),
                    self.info.get("pred_10min_c", 0), self.info.get("pred_30min_c", 0),
                    safe.get("heater", 0), safe.get("vent", 0), safe.get("curtain", 0), safe.get("fan", 0),
                    energy.get("battery_soc_pct", 0), energy.get("battery_power_kw", 0),
                    energy.get("renewable_power_kw", 0), energy.get("grid_power_kw", 0),
                    energy.get("load_power_kw", 0), energy.get("grid_status", "UNKNOWN"),
                    energy.get("unserved_power_kw", 0),
                    json.dumps(self.info.get("safety_overrides", ()), ensure_ascii=False), command_type,
                ]
            )

    def _sac_action(self) -> np.ndarray:
        predicted = self.policy.predict(self.observation, deterministic=True)
        action = predicted[0] if isinstance(predicted, tuple) else predicted
        return np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)

    def _fixed_action(self) -> np.ndarray:
        rh = finite(self.info.get("relative_humidity_pct"), 70.0)
        return strategy_to_normalized(
            {
                "temperature_offset_c": 0.0,
                "comfort_weight": 0.85,
                "energy_weight": 0.50,
                "dehumidify_priority": 0.85 if rh > 88.0 else 0.30,
            }
        )

    def _explain(self, action: np.ndarray) -> Dict[str, Any]:
        current = finite(self.info.get("temperature_c"), 15.0)
        pred30 = finite(self.info.get("pred_30min_c"), current)
        rh = finite(self.info.get("relative_humidity_pct"), 70.0)
        low, high = self.info.get("target_band_c", (8.0, 10.0))
        quality = str(self.info.get("transformer_quality", "UNKNOWN"))
        if quality != "GOOD":
            title = "预测数据质量下降，建议保持安全控制"
            reason = "部分输入数据质量未达到GOOD，本周期不依赖预测值扩大动作。"
        elif pred30 < low:
            title = "预计温度将低于目标，建议保温升温"
            reason = f"当前{current:.1f}℃，预计30分钟后{pred30:.1f}℃，低于下限{low:.1f}℃。"
        elif pred30 > high:
            title = "预计温度将高于目标，建议通风降温"
            reason = f"当前{current:.1f}℃，预计30分钟后{pred30:.1f}℃，高于上限{high:.1f}℃。"
        elif rh > 85.0:
            title = "温度基本稳定，但湿度偏高，建议适度排湿"
            reason = f"当前湿度{rh:.1f}%，需要在避免低温的前提下排湿。"
        else:
            title = "环境处于目标附近，建议保持平稳运行"
            reason = f"预计30分钟后{pred30:.1f}℃，仍处于{low:.1f}–{high:.1f}℃目标附近。"

        preview = StrategyToControlV3(minimum_hold_steps=self.config.minimum_hold_steps)
        preview.reset(self.env._previous_targets)
        plan = preview.plan(action, self.env._controller_state(), self.env._previous_targets)
        safe, overrides = self.env._safety_filter(plan.command_pct)
        return {
            "title": title,
            "reason": reason,
            "strategy": normalized_to_strategy(action),
            "requested_devices_pct": dict(plan.command_pct),
            "safe_devices_pct": dict(safe),
            "planner_constraints": list(plan.constraint_events),
            "safety_overrides": list(overrides),
            "action": action.tolist(),
        }

    def _create_pending(self) -> None:
        action = self._sac_action()
        explanation = self._explain(action)
        self.pending = {
            "decision_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_monotonic": time.monotonic(),
            **explanation,
        }
        self.pending_response = None

    def _process_command(self, command: Mapping[str, Any]) -> None:
        command_type = str(command.get("type", ""))
        self.last_command = {
            "type": command_type,
            "at": datetime.now(timezone.utc).isoformat(),
            "request_id": command.get("request_id"),
        }
        if command_type == "set_mode":
            requested_mode = str(command.get("mode", "")).lower()
            if requested_mode in VALID_MODES:
                self.mode = requested_mode
                self.pending = None
                self.pending_response = None
        elif command_type == "manual_targets":
            targets = command.get("targets", {})
            self.manual_targets = {
                name: float(np.clip(finite(targets.get(name), self.manual_targets[name]), 0.0, 100.0))
                for name in DEVICE_NAMES
            }
            self.total_manual_commands += 1
        elif command_type == "preset":
            name = str(command.get("name", ""))
            if name in self.PRESETS:
                self.manual_targets = dict(self.PRESETS[name])
                self.total_manual_commands += 1
        elif command_type == "ai_decision" and self.pending is not None:
            if command.get("decision_id") == self.pending.get("decision_id"):
                self.pending_response = dict(command)
        elif command_type == "pause":
            self.paused = True
        elif command_type == "resume":
            self.paused = False
        elif command_type == "emergency_stop":
            self.emergency_stop = True
            self.paused = True
            self.pending = None
        elif command_type == "clear_emergency_stop":
            self.emergency_stop = False
            self.paused = False
        elif command_type == "set_grid_status":
            status = str(command.get("status", "AUTO")).upper()
            if self.env._vg is not None and hasattr(self.env._vg, "energy"):
                self.env._vg.energy.set_grid_status_override(status)
                self.grid_event_pending = True
                # 原AI建议基于切换前的能源状态，作废后重新生成更安全。
                self.pending = None
                self.pending_response = None
                self.last_command.update({
                    "status": "ACCEPTED",
                    "requested_grid_status": status,
                    "message": "市电场景命令已接收，将在本控制周期生效",
                })

    def poll_commands(self) -> None:
        for path in sorted(self.command_dir.glob("*.json")):
            try:
                # Windows PowerShell 5.1 的 Set-Content -Encoding UTF8 会写入 BOM。
                # 使用 utf-8-sig 可同时兼容带 BOM 与无 BOM 的前端命令文件。
                command = json.loads(path.read_text(encoding="utf-8-sig"))
                self._process_command(command)
                path.replace(self.archive_dir / path.name)
            except (OSError, json.JSONDecodeError):
                continue

    def _execute(self, action_or_targets: Any, source: str) -> tuple[bool, bool]:
        if source in {"manual", "human_modified", "grid_event"}:
            self.observation, reward, terminated, truncated, self.info = self.env.step_manual_targets(action_or_targets)
        else:
            self.observation, reward, terminated, truncated, self.info = self.env.step(action_or_targets)
        del reward
        overrides = self.info.get("safety_overrides", ())
        self.total_safety_overrides += int(bool(overrides))
        if source.startswith("ai") or source.startswith("human"):
            self.total_ai_steps += 1
        self._log_step(source)
        return bool(terminated), bool(truncated)

    def run_cycle(self) -> tuple[bool, bool, str]:
        if self.emergency_stop or self.paused:
            return False, False, "paused"
        if self.grid_event_pending:
            self.grid_event_pending = False
            # 能源事件发生时维持上一周期设备目标，只改变供电边界。
            targets = dict(getattr(self.env, "_previous_targets", self.manual_targets))
            terminated, truncated = self._execute(targets, "grid_event")
            return terminated, truncated, "grid_event"
        if self.mode == "manual":
            terminated, truncated = self._execute(self.manual_targets, "manual")
            return terminated, truncated, "manual"
        if self.mode == "auto":
            action = self._sac_action()
            terminated, truncated = self._execute(action, "ai_auto")
            self.pending = self._explain(action)
            return terminated, truncated, "ai_auto"

        # AI辅助：先发布建议，等待人工确认，不在等待期间偷偷推进仿真。
        if self.pending is None:
            self._create_pending()
            return False, False, "waiting_human"
        if self.pending_response is None:
            if time.monotonic() - finite(self.pending.get("created_monotonic")) > self.config.assist_timeout_seconds:
                terminated, truncated = self._execute(self._fixed_action(), "human_timeout_fixed")
                self.pending = None
                return terminated, truncated, "human_timeout_fixed"
            return False, False, "waiting_human"

        response = self.pending_response
        decision = str(response.get("decision", "reject")).lower()
        if decision == "approve":
            action = np.asarray(self.pending["action"], dtype=np.float32)
            terminated, truncated = self._execute(action, "human_approved_ai")
            source = "human_approved_ai"
        elif decision == "modify":
            targets = response.get("targets", self.manual_targets)
            terminated, truncated = self._execute(targets, "human_modified")
            source = "human_modified"
        else:
            terminated, truncated = self._execute(self._fixed_action(), "human_rejected_fixed")
            source = "human_rejected_fixed"
        self.pending = None
        self.pending_response = None
        return terminated, truncated, source

    def _append_trend(self) -> None:
        low, high = self.info.get("target_band_c", (0.0, 0.0))
        energy = self.info.get("energy", {})
        self.trend.append(
            {
                "step": int(self.info.get("decision_step", 0)),
                "temperature_c": finite(self.info.get("temperature_c")),
                "pred_10min_c": finite(self.info.get("pred_10min_c")),
                "pred_30min_c": finite(self.info.get("pred_30min_c")),
                "target_low_c": finite(low),
                "target_high_c": finite(high),
                "battery_soc_pct": finite(energy.get("battery_soc_pct")),
                "battery_power_kw": finite(energy.get("battery_power_kw")),
                "renewable_power_kw": finite(energy.get("renewable_power_kw")),
                "grid_power_kw": finite(energy.get("grid_power_kw")),
                "load_power_kw": finite(energy.get("load_power_kw")),
                "unserved_power_kw": finite(energy.get("unserved_power_kw")),
            }
        )

    def publish(self, status: str, control_source: str) -> None:
        record = dict(getattr(self.env, "_latest_record", {}) or {})
        requested = dict(self.info.get("requested_targets_pct", {}))
        safe = dict(self.info.get("safe_targets_pct", {}))
        actual = dict(self.info.get("actuals_pct", {}))
        if not self.trend or self.trend[-1]["step"] != int(self.info.get("decision_step", 0)):
            self._append_trend()
        devices = {
            name: {
                "requested_pct": finite(requested.get(name)),
                "safe_pct": finite(safe.get(name)),
                "actual_pct": finite(actual.get(name)),
            }
            for name in DEVICE_NAMES
        }
        pending_public = None
        if self.pending is not None:
            pending_public = {key: value for key, value in self.pending.items() if key not in {"created_monotonic", "action"}}
        energy = dict(self.info.get("energy", {}) or {})
        payload = {
            "schema_version": "4.2.1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "status": status,
                "source": self.config.source,
                "mode": self.mode,
                "paused": self.paused,
                "emergency_stop": self.emergency_stop,
                "control_source": control_source,
                "control_step": int(self.info.get("decision_step", 0)),
                "sim_hour": finite(self.info.get("sim_hour")),
                "greenhouse_name": self.config.greenhouse_name,
                "crop": self.config.crop,
            },
            "environment": {
                "timestamp": record.get("timestamp"),
                "weather_scene": record.get("weather_scene", record.get("scene", "--")),
                "is_day": bool(record.get("is_day", 0)),
                "indoor": {
                    "east_temperature_c": finite(record.get("t_east_c")),
                    "center_temperature_c": finite(record.get("t_center_c")),
                    "west_temperature_c": finite(record.get("t_west_c")),
                    "relative_humidity_pct": finite(record.get("rh_center_pct")),
                    "co2_ppm": finite(record.get("co2_ppm")),
                    "canopy_temperature_c": finite(record.get("canopy_temp_c")),
                    "substrate_temperature_c": finite(record.get("substrate_temp_c")),
                    "substrate_moisture_pct": finite(record.get("substrate_moisture_pct")),
                },
                "outdoor": {
                    "temperature_c": finite(record.get("out_temp_c")),
                    "relative_humidity_pct": finite(record.get("out_rh_pct")),
                    "solar_w_m2": finite(record.get("out_solar_w_m2")),
                    "wind_m_s": finite(record.get("out_wind_m_s")),
                },
                "quality": {key: value for key, value in record.items() if key.endswith("_quality")},
            },
            "prediction": {
                "status": self.info.get("transformer_status", "UNKNOWN"),
                "quality": self.info.get("transformer_quality", "UNKNOWN"),
                "temperature_10min_c": finite(self.info.get("pred_10min_c")),
                "temperature_30min_c": finite(self.info.get("pred_30min_c")),
                "target_band_c": list(self.info.get("target_band_c", (0.0, 0.0))),
            },
            "devices": devices,
            "energy": {
                "enabled": bool(getattr(self.env._cfg, "energy_enabled", False)),
                "ems_mode": "SELF_CONSUMPTION_AND_PEAK_SHAVING",
                "battery_soc_pct": finite(energy.get("battery_soc_pct")),
                "battery_power_kw": finite(energy.get("battery_power_kw")),
                "pv_power_kw": finite(energy.get("pv_power_kw")),
                "wind_power_kw": finite(energy.get("wind_power_kw")),
                "renewable_power_kw": finite(energy.get("renewable_power_kw")),
                "grid_power_kw": finite(energy.get("grid_power_kw")),
                "load_power_kw": finite(energy.get("load_power_kw")),
                "served_power_kw": finite(energy.get("served_power_kw")),
                "unserved_power_kw": finite(energy.get("unserved_power_kw")),
                "renewable_curtailed_kw": finite(energy.get("renewable_curtailed_kw")),
                "power_supply_ratio": finite(energy.get("power_supply_ratio"), 1.0),
                "grid_status": str(energy.get("grid_status", "UNKNOWN")),
                "grid_status_override": getattr(self.env._vg.energy, "grid_status_override", None),
                "cumulative_grid_kwh": finite(energy.get("cumulative_grid_kwh")),
                "cumulative_renewable_used_kwh": finite(energy.get("cumulative_renewable_used_kwh")),
                "cumulative_unserved_kwh": finite(energy.get("cumulative_unserved_kwh")),
                "boundary": "当前由规则EMS调度储能；SAC V3的节能权重仅影响环境设备策略，尚未直接输出储能功率。",
            },
            "manual_targets_pct": dict(self.manual_targets),
            "ai": {
                "pending": pending_public,
                "technical_strategy": self.info.get("strategy_action"),
                "planner_constraints": list(self.info.get("planner_constraint_events", ())),
            },
            "safety": {
                "current_overrides": list(self.info.get("safety_overrides", ())),
                "total_override_steps": self.total_safety_overrides,
            },
            "metrics": {
                "manual_command_count": self.total_manual_commands,
                "ai_executed_steps": self.total_ai_steps,
            },
            "last_command": self.last_command,
            "trend": list(self.trend),
        }
        self._last_payload = payload
        atomic_json(self.state_file, payload)

    def run(self) -> None:
        status = "RUNNING"
        control_source = "initialization"
        try:
            while not self.stop_file.exists():
                started = time.monotonic()
                self.poll_commands()
                terminated, truncated, control_source = self.run_cycle()
                self.publish(status=status, control_source=control_source)
                if terminated or truncated:
                    status = "COMPLETED"
                    self.publish(status=status, control_source=control_source)
                    break
                if self.config.max_control_steps > 0 and int(self.info.get("decision_step", 0)) >= self.config.max_control_steps:
                    status = "COMPLETED"
                    self.publish(status=status, control_source=control_source)
                    break
                elapsed = time.monotonic() - started
                time.sleep(max(0.05, self.config.wall_interval_seconds - elapsed))
        finally:
            self.env.close()
            if status == "RUNNING":
                status = "STOPPED"
            if self._last_payload:
                self._last_payload["runtime"]["status"] = status
                self._last_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                atomic_json(self.state_file, self._last_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绿风润耕V4统一交互运行层")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ui_demo_config.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = UIConfig.load(args.config.resolve())
    config.validate()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "OK",
                    "source": config.source,
                    "default_mode": config.default_mode,
                    "state_root": str(config.state_root),
                    "models": {
                        "transformer": str(config.transformer_model_path),
                        "sac": str(config.sac_model_path),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    runtime = InteractiveRuntime(config)
    runtime.run()


if __name__ == "__main__":
    main()
