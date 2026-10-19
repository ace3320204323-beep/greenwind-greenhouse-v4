"""配置加载与默认值。

从场景 JSON 加载配置，所有目标值、设备参数、天气参数均配置化，不写死在代码中。
缺省字段回退到本文件的默认值，保证最小可用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .schemas import CropStage, ZoneId


# ---------------------------------------------------------------------------
# 生长阶段 → 目标温湿度预设
# ---------------------------------------------------------------------------

# 依据设施草莓农业技术指导：
#   采果期/结果期（fruiting/harvest）：白天 20~25℃、夜间 8~10℃
#   开花期（flowering）：白天 22~25℃、夜间 8~10℃
GROWTH_STAGE_TARGETS = {
    "fruiting": {"day_min": 20.0, "day_max": 25.0, "night_min": 8.0, "night_max": 10.0},
    "harvest": {"day_min": 20.0, "day_max": 25.0, "night_min": 8.0, "night_max": 10.0},
    "flowering": {"day_min": 22.0, "day_max": 25.0, "night_min": 8.0, "night_max": 10.0},
}


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------


@dataclass
class CropTargets:
    """某生育阶段的目标温湿度范围（℃ / %）。"""
    day_temp_min_c: float = 20.0
    day_temp_max_c: float = 25.0
    night_temp_min_c: float = 8.0
    night_temp_max_c: float = 10.0
    humidity_min_pct: float = 50.0
    humidity_max_pct: float = 70.0
    # 白天/夜晚切换的太阳辐射阈值 (W/m²)，高于此视为白天
    day_radiation_threshold_w_m2: float = 50.0


@dataclass
class ZoneParams:
    """单个区域的热物性参数。三区可有细微差别。"""
    # 本区空气的基础热容 C_air (kJ/K)。1800 m³ 棚体三等分时
    # 约为 724 kJ/K。实际有效热容 = 本值 × thermal_inertia。
    air_heat_capacity_kj_k: float = 723.6
    vent_exposure: float = 1.0                  # 受通风影响权重（近风口区域更大）
    thermal_inertia: float = 1.0                # 有效热容修正系数（中区更大）


@dataclass
class PlantParams:
    """棚体与作物物理参数。"""
    cover_area_m2: float = 600.0
    volume_m3: float = 1800.0
    air_density_kg_m3: float = 1.2
    air_heat_capacity_j_kg_k: float = 1005.0
    # 冠层 / 基质热物性
    canopy_heat_capacity_kj_k: float = 200.0
    substrate_heat_capacity_kj_k: float = 1500.0
    structure_heat_capacity_kj_k: float = 2500.0
    canopy_air_exchange_kw_k: float = 0.15      # 冠层-空气显热交换系数 (kW/K)
    substrate_air_exchange_kw_k: float = 0.06   # 基质-空气显热交换系数 (kW/K)
    structure_air_exchange_kw_k: float = 0.25   # 棚架/管路等蓄热体-空气交换 (kW/K)
    zone_neighbor_exchange_kw_k: float = 0.015  # 相邻分区空气换热 (kW/K)
    # 蒸腾
    leaf_area_index: float = 2.5                # 叶面积指数，随生育期漂移
    transpiration_base_g_s: float = 0.02
    substrate_evaporation_base: float = 0.004


@dataclass
class DeviceSpec:
    """单个设备的动态参数。"""
    device_id: str = ""
    kind: str = ""
    unit: str = "%"
    action_type: str = "SET_POSITION"
    max_rate_per_s: float = 5.0         # 最大动作速度（%/s 或单位/s）
    deadband: float = 0.5               # 死区
    hysteresis: float = 1.0             # 回差
    min_run_seconds: float = 0.0        # 最小启停时间
    min_stop_seconds: float = 0.0
    rated_power_kw: float = 0.0         # 额定功率
    stuck_probability: float = 0.0      # 卡涩概率（每次动作尝试）


@dataclass
class WeatherParams:
    """天气模型参数。"""
    base_outdoor_temp_c: float = 0.0
    daily_temp_amplitude_c: float = 9.0
    base_solar_irradiance_w_m2: float = 850.0
    base_humidity_pct: float = 70.0
    base_wind_speed_m_s: float = 2.0
    # 一阶相关扰动系数 alpha（0~1）
    weather_noise_alpha: float = 0.8
    weather_noise_sigma_temp: float = 0.3
    weather_noise_sigma_solar: float = 20.0


@dataclass
class SensorParams:
    """传感器噪声与误差参数。"""
    temperature_noise_sigma_c: float = 0.15
    humidity_noise_sigma_pct: float = 0.8
    radiation_noise_sigma: float = 15.0
    wind_noise_sigma: float = 0.2
    temperature_bias_c: float = 0.0         # 固定偏置
    humidity_bias_pct: float = 0.0
    temperature_quantization_c: float = 0.1  # 量化步长
    humidity_quantization_pct: float = 0.5
    drift_rate_c_per_hour: float = 0.0       # 缓慢漂移速率
    packet_loss_probability: float = 0.0     # 丢包概率


@dataclass
class SimulationParams:
    """仿真与积分参数。"""
    dt_seconds: float = 60.0            # 物理积分步长（仿真秒）
    duration_hours: float = 72.0
    realtime_factor: float = 0.0        # 0=非实时全速，60=60倍加速
    sim_start: str = "2026-01-10T08:00:00+08:00"


@dataclass
class EnergyParams:
    """单棚能源子系统参数。

    符号约定：battery_power_kw > 0 表示储能放电，< 0 表示充电。
    本参数组是仿真初值，真实接入时必须按电池、PCS、光伏/风机和并网点重新标定。
    """
    battery_capacity_kwh: float = 100.0
    initial_soc_pct: float = 70.0
    min_soc_pct: float = 15.0
    max_soc_pct: float = 95.0
    max_charge_power_kw: float = 30.0
    max_discharge_power_kw: float = 30.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    pv_peak_power_kw: float = 20.0
    wind_rated_power_kw: float = 5.0
    wind_cut_in_m_s: float = 2.0
    wind_rated_speed_m_s: float = 10.0
    wind_cut_out_m_s: float = 25.0
    inverter_efficiency: float = 0.96
    grid_max_import_kw: float = 50.0
    grid_limited_import_kw: float = 10.0
    peak_shaving_threshold_kw: float = 20.0
    base_critical_load_kw: float = 0.6
    allow_grid_charge: bool = False


@dataclass
class GreenhouseConfig:
    """顶层配置。"""
    scenario_id: str = "shandong_winter_strawberry"
    greenhouse_id: str = "GH001"
    crop_type: str = "STRAWBERRY"
    crop_stage: CropStage = CropStage.FRUIT_EXPANSION
    # 生长阶段：fruiting / harvest / flowering，驱动目标温湿度预设
    growth_stage: str = "fruiting"
    targets: CropTargets = field(default_factory=CropTargets)
    zones: dict[str, ZoneParams] = field(default_factory=lambda: {
        ZoneId.EAST.value: ZoneParams(air_heat_capacity_kj_k=723.6, vent_exposure=1.25, thermal_inertia=0.9),
        ZoneId.CENTER.value: ZoneParams(air_heat_capacity_kj_k=723.6, vent_exposure=0.8, thermal_inertia=1.2),
        ZoneId.WEST.value: ZoneParams(air_heat_capacity_kj_k=723.6, vent_exposure=1.05, thermal_inertia=1.0),
    })
    plant: PlantParams = field(default_factory=PlantParams)
    devices: list[DeviceSpec] = field(default_factory=list)
    weather: WeatherParams = field(default_factory=WeatherParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    energy: EnergyParams = field(default_factory=EnergyParams)
    energy_enabled: bool = True
    # 场景天气时间表（阶段列表）
    weather_schedule: list[dict[str, Any]] = field(default_factory=list)
    # 故障计划
    fault_schedule: list[dict[str, Any]] = field(default_factory=list)
    # 市电状态时间表：NORMAL / LIMITED / OUTAGE
    grid_schedule: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 默认设备清单（任务书 2.1 基础设备）
# ---------------------------------------------------------------------------

DEFAULT_DEVICES = [
    DeviceSpec("vent_01", "VENT", "%", "SET_POSITION",
               max_rate_per_s=1.0, deadband=0.5, hysteresis=1.0,
               min_stop_seconds=30.0, rated_power_kw=0.3, stuck_probability=0.0),
    DeviceSpec("fan_01", "FAN", "%", "SET_POWER",
               max_rate_per_s=20.0, deadband=0.5, hysteresis=1.0,
               min_run_seconds=20.0, min_stop_seconds=20.0,
               rated_power_kw=1.5, stuck_probability=0.0),
    DeviceSpec("curtain_01", "CURTAIN", "%", "SET_POSITION",
               max_rate_per_s=0.8, deadband=0.5, hysteresis=1.0,
               min_stop_seconds=60.0, rated_power_kw=0.5, stuck_probability=0.0),
    DeviceSpec("heater_01", "HEATER", "%", "SET_POWER",
               max_rate_per_s=10.0, deadband=0.5, hysteresis=1.0,
               min_run_seconds=30.0, rated_power_kw=30.0, stuck_probability=0.0),
    DeviceSpec("irrigation_01", "IRRIGATION", "L/min", "SET_FLOW",
               max_rate_per_s=2.0, deadband=0.05, hysteresis=0.1,
               rated_power_kw=1.0, stuck_probability=0.0),
]


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------


def _deep_update(base: Any, override: dict[str, Any]) -> Any:
    """递归合并 override 到 base（仅覆盖提供的键）。"""
    if not isinstance(override, dict):
        return override
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(getattr(base, k, None), object) and hasattr(base, k):
            cur = getattr(base, k)
            if hasattr(cur, "__dataclass_fields__"):
                _deep_update(cur, v)
            else:
                setattr(base, k, v)
        elif isinstance(v, list):
            setattr(base, k, v)
        else:
            setattr(base, k, v)
    return base


def _build_from_dataclass(cls: Any, data: dict[str, Any]) -> Any:
    obj = cls()
    _deep_update(obj, data)
    return obj


def load_config(path: str | Path) -> GreenhouseConfig:
    """加载场景 JSON 到 GreenhouseConfig。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = GreenhouseConfig()

    # 顶层标量
    for key in ("scenario_id", "greenhouse_id", "crop_type", "energy_enabled"):
        if key in raw:
            setattr(cfg, key, raw[key])

    if "crop_stage" in raw:
        cfg.crop_stage = CropStage(raw["crop_stage"])

    # 生长阶段：驱动目标预设（若场景未显式指定 targets）
    if "growth_stage" in raw:
        cfg.growth_stage = raw["growth_stage"]

    # 目标温度：优先显式 targets；否则按 growth_stage 预设
    if "targets" in raw:
        cfg.targets = _build_from_dataclass(CropTargets, raw["targets"])
    else:
        preset = GROWTH_STAGE_TARGETS.get(cfg.growth_stage,
                                          GROWTH_STAGE_TARGETS["fruiting"])
        cfg.targets = CropTargets(
            day_temp_min_c=preset["day_min"], day_temp_max_c=preset["day_max"],
            night_temp_min_c=preset["night_min"], night_temp_max_c=preset["night_max"],
        )

    if "plant" in raw:
        plant_raw = dict(raw["plant"])
        # 兼容 P0/P0.1 旧场景字段；加载后统一转成带单位的新字段。
        aliases = {
            "canopy_air_exchange": "canopy_air_exchange_kw_k",
            "substrate_air_exchange": "substrate_air_exchange_kw_k",
        }
        for old_key, new_key in aliases.items():
            if old_key in plant_raw and new_key not in plant_raw:
                plant_raw[new_key] = plant_raw.pop(old_key)
        cfg.plant = _build_from_dataclass(PlantParams, plant_raw)
    if "weather" in raw:
        cfg.weather = _build_from_dataclass(WeatherParams, raw["weather"])
    if "sensor" in raw:
        cfg.sensor = _build_from_dataclass(SensorParams, raw["sensor"])
    if "simulation" in raw:
        cfg.simulation = _build_from_dataclass(SimulationParams, raw["simulation"])
    if "energy" in raw:
        cfg.energy = _build_from_dataclass(EnergyParams, raw["energy"])

    if "zones" in raw:
        cfg.zones = {
            zid: _build_from_dataclass(ZoneParams, zraw)
            for zid, zraw in raw["zones"].items()
        }

    # 设备：优先场景自定义，否则用默认清单
    if "devices" in raw:
        cfg.devices = [
            _build_from_dataclass(DeviceSpec, d) for d in raw["devices"]
        ]
    if not cfg.devices:
        cfg.devices = list(DEFAULT_DEVICES)

    cfg.weather_schedule = raw.get("weather_schedule", [])
    cfg.fault_schedule = raw.get("fault_schedule", [])
    cfg.grid_schedule = raw.get("grid_schedule", [])

    return cfg


def config_fingerprint(cfg: GreenhouseConfig) -> str:
    """生成配置的稳定哈希，写入 manifest。"""
    import hashlib
    payload = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
