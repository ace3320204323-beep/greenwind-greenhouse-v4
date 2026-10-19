"""统一字段字典、枚举与数据契约。

本模块是整个虚拟大棚的"字段契约"来源。任务书第五章、第六章、第七章
规定的字段、枚举、单位与命令/反馈结构均在此固定，其它模块只引用、不重复定义。

设计约束：
- 只用 Python 标准库（dataclasses、enum、datetime、uuid）。
- 不依赖、不引用主包 greenwind_edge（虚拟大棚完全独立）。
- 真实状态 TrueState 与观测 Observation 严格分离，控制器只允许读取 Observation。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

# 时区：任务书要求 ISO 8601 且带 +08:00
CN_TZ = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class Quality(str, Enum):
    """观测数据质量状态（任务书 4.2）。"""
    GOOD = "GOOD"
    SUSPECT = "SUSPECT"
    MISSING = "MISSING"
    STALE = "STALE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    STUCK = "STUCK"
    SPIKE = "SPIKE"
    DRIFT = "DRIFT"
    DISCONNECTED = "DISCONNECTED"


class ZoneId(str, Enum):
    EAST = "EAST"
    CENTER = "CENTER"
    WEST = "WEST"
    GLOBAL = "GLOBAL"


class CropStage(str, Enum):
    FLOWERING = "FLOWERING"
    FRUIT_EXPANSION = "FRUIT_EXPANSION"
    RIPENING = "RIPENING"


class ControlMode(str, Enum):
    MANUAL = "MANUAL"
    RULE = "RULE"
    FIXED_MODEL = "FIXED_MODEL"
    ADAPTIVE_MODEL = "ADAPTIVE_MODEL"
    SAFE_FALLBACK = "SAFE_FALLBACK"


class ActionType(str, Enum):
    SET_POSITION = "SET_POSITION"
    SET_POWER = "SET_POWER"
    SET_FLOW = "SET_FLOW"
    START = "START"
    STOP = "STOP"
    RESET = "RESET"


class CommandSource(str, Enum):
    LOCAL_CONTROLLER = "LOCAL_CONTROLLER"
    MANUAL_OPERATOR = "MANUAL_OPERATOR"
    SAFETY_ENGINE = "SAFETY_ENGINE"


class DeviceState(str, Enum):
    OFF = "OFF"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FAULT = "FAULT"


class FeedbackStatus(str, Enum):
    RECEIVED = "RECEIVED"
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class SafetyResult(str, Enum):
    PASS = "PASS"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class DeviceKind(str, Enum):
    VENT = "VENT"
    FAN = "FAN"
    CURTAIN = "CURTAIN"
    HEATER = "HEATER"
    IRRIGATION = "IRRIGATION"


# 设备类型到命令/反馈的映射
DEVICE_DEFAULTS = {
    DeviceKind.VENT: {"unit": "%", "action": ActionType.SET_POSITION},
    DeviceKind.FAN: {"unit": "%", "action": ActionType.SET_POWER},
    DeviceKind.CURTAIN: {"unit": "%", "action": ActionType.SET_POSITION},
    DeviceKind.HEATER: {"unit": "%", "action": ActionType.SET_POWER},
    DeviceKind.IRRIGATION: {"unit": "L/min", "action": ActionType.SET_FLOW},
}

# 命令优先级（任务书第六章），数值越小优先级越高
COMMAND_PRIORITY = {
    "EMERGENCY_STOP": 0,
    "MECHANICAL_PROTECT": 1,
    "MANUAL": 10,
    "SAFETY_ENGINE": 20,
    "CONTROL": 30,
    "CLOUD_SUGGEST": 40,
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).astimezone(CN_TZ).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 基础标识（任务书 5.1）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunIdentity:
    run_id: str
    scenario_id: str
    seed: int
    greenhouse_id: str = "GH001"
    crop_type: str = "STRAWBERRY"
    crop_stage: CropStage = CropStage.FLOWERING
    software_version: str = "virtual-greenhouse-p0-1.0.0"


# ---------------------------------------------------------------------------
# 真实状态（任务书 4.1）：仅存于模拟器内部，控制器不得读取
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrueState:
    sim_time: str = ""
    sim_step: int = 0
    dt_seconds: float = 0.0

    # 三区空气温度 (℃)
    zone_east_air_temperature_c: float = 22.0
    zone_center_air_temperature_c: float = 22.0
    zone_west_air_temperature_c: float = 22.0
    # 三区绝对湿度 (g/m³)
    zone_east_absolute_humidity_g_m3: float = 8.0
    zone_center_absolute_humidity_g_m3: float = 8.0
    zone_west_absolute_humidity_g_m3: float = 8.0

    canopy_temperature_c: float = 21.0
    canopy_transpiration_g_s: float = 0.0
    # 结露量（g/m³/s），超饱和水汽转化为结露并从空气扣除（质量守恒）
    condensation_rate_g_m3_s: float = 0.0
    substrate_temperature_c: float = 20.0
    # 棚架、管路及棚内设施的等效蓄热温度（仅供仿真评价）
    structure_temperature_c: float = 19.0
    substrate_moisture_pct: float = 60.0
    cover_temperature_c: float = 18.0
    indoor_co2_ppm: float = 450.0

    # 设备实际状态
    actual_vent_position_pct: float = 0.0
    actual_fan_power_pct: float = 0.0
    actual_curtain_position_pct: float = 0.0
    actual_heater_power_pct: float = 0.0
    actual_irrigation_flow_l_min: float = 0.0

    # 能源子系统真值（观测端将另行输出带质量标记的能源测点）
    battery_soc_pct: float = 70.0
    battery_power_kw: float = 0.0
    pv_power_kw: float = 0.0
    wind_power_kw: float = 0.0
    renewable_power_kw: float = 0.0
    grid_power_kw: float = 0.0
    load_power_kw: float = 0.0
    unserved_power_kw: float = 0.0
    renewable_curtailed_kw: float = 0.0
    power_supply_ratio: float = 1.0
    grid_status: str = "NORMAL"

    # 真实物理参数（仅模拟器计算与事后评价，不发送控制器）
    true_heat_transfer_coefficient: float = 5.5
    true_ventilation_coefficient: float = 0.12
    true_solar_gain_coefficient: float = 0.0005
    true_transpiration_coefficient: float = 0.02
    true_air_leakage_rate: float = 0.003

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 观测（任务书 4.2）：控制器唯一可读数据
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Observation:
    point_id: str
    greenhouse_id: str
    zone_id: str
    metric: str
    value: float | None
    unit: str
    observed_at: str
    received_at: str
    quality: Quality = Quality.GOOD
    source: str = "SIMULATOR"
    sequence: int = 0
    run_id: str = ""
    scenario_id: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quality"] = self.quality.value
        return d


@dataclass(slots=True)
class ObservationBatch:
    observations: list[Observation]
    sim_time: str
    sim_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "sim_step": self.sim_step,
            "observations": [o.to_dict() for o in self.observations],
        }


# ---------------------------------------------------------------------------
# 控制指令（任务书第六章）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ControlCommand:
    command_id: str
    trace_id: str
    greenhouse_id: str
    device_id: str
    action_type: ActionType
    target_value: float
    unit: str
    issued_at: str
    effective_at: str
    expires_at: str
    control_mode: ControlMode = ControlMode.RULE
    source: CommandSource = CommandSource.LOCAL_CONTROLLER
    priority: int = COMMAND_PRIORITY["CONTROL"]
    model_version: str = ""
    safety_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        d["control_mode"] = self.control_mode.value
        d["source"] = self.source.value
        return d


# ---------------------------------------------------------------------------
# 设备反馈（任务书第七章）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CommandFeedback:
    feedback_id: str
    command_id: str
    trace_id: str
    device_id: str
    status: FeedbackStatus
    target_value: float
    actual_value: float
    unit: str
    feedback_at: str
    fault_code: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# 事件与 step 结果
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Event:
    event_id: str
    event_type: str
    sim_time: str
    sim_step: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detail"] = json_safe(self.detail)
        return d


def json_safe(obj: Any) -> Any:
    """把非 JSON 原生类型（枚举等）安全转为可序列化值。"""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
