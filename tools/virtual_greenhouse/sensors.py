"""传感器模型。

把真实状态转换为带噪声、偏置、量化、漂移、丢包和故障的观测值（任务书 4.2 / 11 / 12）。
观测是控制器唯一可读数据，必须与真实状态分离。
"""

from __future__ import annotations

import math
import random

from .config import SensorParams
from .schemas import Quality, Observation, ZoneId


# 需要观测的 metric -> (真值来源函数名, 单位)
# 温度类按三区分测点；其它环境量取全局/代表性测点。
def _zone_metrics() -> list[tuple[str, str]]:
    return [
        ("air_temperature_c", "°C"),
        ("relative_humidity_pct", "%"),
        ("absolute_humidity_g_m3", "g/m³"),
    ]


class SensorModel:
    def __init__(self, params: SensorParams, seed: int):
        self.params = params
        self.rng = random.Random(seed ^ 0x5F3759DF)  # 独立随机流
        # 漂移累积状态（每个测点）
        self._drift: dict[str, float] = {}
        # 故障注入状态
        self._stuck_points: set[str] = set()
        self._stuck_values: dict[str, float] = {}
        self._drift_points: dict[str, float] = {}   # point_id -> 额外漂移速率
        self._spike_points: dict[str, float] = {}   # point_id -> 跳变时刻
        self._disconnected_points: set[str] = set()

    # 测点 ID 生成
    @staticmethod
    def point_id(zone: str, metric: str) -> str:
        return f"{zone.lower()}_{metric}"

    def _quantize(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return round(value / step) * step

    def _apply_fault(self, pid: str, value: float, metric: str, sim_seconds: float,
                     dt_seconds: float) -> tuple[float, Quality, bool]:
        """应用故障，返回 (修正值, 质量, 是否丢失)。"""
        q = Quality.GOOD

        # 丢包：按概率返回缺失
        if self.rng.random() < self.params.packet_loss_probability:
            return None, Quality.MISSING, True

        if pid in self._disconnected_points:
            return None, Quality.DISCONNECTED, True

        if pid in self._stuck_points:
            # 卡死：保持最后值
            v = self._stuck_values.get(pid, value)
            self._stuck_values[pid] = v
            return v, Quality.STUCK, False

        if pid in self._drift_points:
            # 漂移：累积额外偏差
            self._drift[pid] = self._drift.get(pid, 0.0) + self._drift_points[pid] * dt_seconds / 3600.0
            value = value + self._drift[pid]
            q = Quality.DRIFT

        if pid in self._spike_points:
            # 跳变：一次性的不合理值
            q = Quality.SPIKE
            value = value * 3.0 if value >= 0 else value - 10.0
            del self._spike_points[pid]

        return value, q, False

    def observe(self, truth, weather, sim_time: str, sim_step: int,
                sim_seconds: float, dt_seconds: float,
                run_id: str, scenario_id: str, seed: int, greenhouse_id: str,
                sequence: int) -> list[Observation]:
        """根据真实状态与天气生成观测列表。"""
        p = self.params
        obs: list[Observation] = []

        # 三区空气温度 / 湿度
        zone_temp = {
            "EAST": truth.zone_east_air_temperature_c,
            "CENTER": truth.zone_center_air_temperature_c,
            "WEST": truth.zone_west_air_temperature_c,
        }
        zone_abs_hum = {
            "EAST": truth.zone_east_absolute_humidity_g_m3,
            "CENTER": truth.zone_center_absolute_humidity_g_m3,
            "WEST": truth.zone_west_absolute_humidity_g_m3,
        }

        def _rh(abs_h: float, temp: float) -> float:
            es = 6.112 * math.exp(17.62 * temp / (243.12 + temp))
            sat = 216.7 * es / (273.15 + temp)
            return min(100.0, max(0.0, abs_h / sat * 100.0)) if sat > 0 else 100.0

        for zone in ("EAST", "CENTER", "WEST"):
            t_true = zone_temp[zone]
            abs_h = zone_abs_hum[zone]
            rh_true = _rh(abs_h, t_true)

            # 温度测点
            t_val = t_true + self.rng.gauss(0.0, p.temperature_noise_sigma_c) + p.temperature_bias_c
            t_val = self._quantize(t_val, p.temperature_quantization_c)
            pid = self.point_id(zone, "air_temperature_c")
            t_val, t_q, t_drop = self._apply_fault(pid, t_val, "temp", sim_seconds, dt_seconds)
            obs.append(Observation(
                point_id=pid, greenhouse_id=greenhouse_id, zone_id=zone,
                metric="air_temperature_c", value=round(t_val, 3) if t_val is not None else None,
                unit="°C", observed_at=sim_time, received_at=sim_time,
                quality=t_q, sequence=sequence, run_id=run_id,
                scenario_id=scenario_id, seed=seed,
            ))

            # 湿度测点
            rh_val = rh_true + self.rng.gauss(0.0, p.humidity_noise_sigma_pct) + p.humidity_bias_pct
            rh_val = self._quantize(rh_val, p.humidity_quantization_pct)
            pid = self.point_id(zone, "relative_humidity_pct")
            rh_val, rh_q, rh_drop = self._apply_fault(pid, rh_val, "rh", sim_seconds, dt_seconds)
            obs.append(Observation(
                point_id=pid, greenhouse_id=greenhouse_id, zone_id=zone,
                metric="relative_humidity_pct", value=round(rh_val, 3) if rh_val is not None else None,
                unit="%", observed_at=sim_time, received_at=sim_time,
                quality=rh_q, sequence=sequence, run_id=run_id,
                scenario_id=scenario_id, seed=seed,
            ))

        # 冠层温度（CENTER 区代表）
        obs.append(Observation(
            point_id="center_canopy_temperature_c", greenhouse_id=greenhouse_id,
            zone_id="CENTER", metric="canopy_temperature_c",
            value=round(truth.canopy_temperature_c + self.rng.gauss(0.0, p.temperature_noise_sigma_c), 3),
            unit="°C", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))

        # 基质温度 / 水分
        obs.append(Observation(
            point_id="center_substrate_temperature_c", greenhouse_id=greenhouse_id,
            zone_id="CENTER", metric="substrate_temperature_c",
            value=round(truth.substrate_temperature_c + self.rng.gauss(0.0, p.temperature_noise_sigma_c), 3),
            unit="°C", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))
        obs.append(Observation(
            point_id="center_substrate_moisture_pct", greenhouse_id=greenhouse_id,
            zone_id="CENTER", metric="substrate_moisture_pct",
            value=round(truth.substrate_moisture_pct + self.rng.gauss(0.0, 0.3), 3),
            unit="%", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))

        # CO2
        obs.append(Observation(
            point_id="center_co2_ppm", greenhouse_id=greenhouse_id,
            zone_id="CENTER", metric="co2_ppm",
            value=round(truth.indoor_co2_ppm + self.rng.gauss(0.0, 10.0), 2),
            unit="ppm", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))

        # 室外边界（传感器测点，含噪声）
        obs.append(Observation(
            point_id="outdoor_air_temperature_c", greenhouse_id=greenhouse_id,
            zone_id="GLOBAL", metric="outdoor_air_temperature_c",
            value=round(weather.outdoor_air_temperature_c + self.rng.gauss(0.0, p.temperature_noise_sigma_c), 3),
            unit="°C", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))
        obs.append(Observation(
            point_id="outdoor_relative_humidity_pct", greenhouse_id=greenhouse_id,
            zone_id="GLOBAL", metric="outdoor_relative_humidity_pct",
            value=round(weather.outdoor_relative_humidity_pct + self.rng.gauss(0.0, p.humidity_noise_sigma_pct), 3),
            unit="%", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))
        obs.append(Observation(
            point_id="outdoor_solar_irradiance_w_m2", greenhouse_id=greenhouse_id,
            zone_id="GLOBAL", metric="outdoor_solar_irradiance_w_m2",
            value=round(weather.outdoor_solar_irradiance_w_m2 + self.rng.gauss(0.0, p.radiation_noise_sigma), 3),
            unit="W/m²", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))
        obs.append(Observation(
            point_id="outdoor_wind_speed_m_s", greenhouse_id=greenhouse_id,
            zone_id="GLOBAL", metric="outdoor_wind_speed_m_s",
            value=round(weather.outdoor_wind_speed_m_s + self.rng.gauss(0.0, p.wind_noise_sigma), 3),
            unit="m/s", observed_at=sim_time, received_at=sim_time,
            quality=Quality.GOOD, sequence=sequence, run_id=run_id,
            scenario_id=scenario_id, seed=seed,
        ))

        # 能源测点：这些值是虚拟电表/BMS/PCS的可观测量，不包含未来信息。
        energy_points = (
            ("battery_soc_pct", truth.battery_soc_pct, "%"),
            ("battery_power_kw", truth.battery_power_kw, "kW"),
            ("pv_power_kw", truth.pv_power_kw, "kW"),
            ("wind_power_kw", truth.wind_power_kw, "kW"),
            ("renewable_power_kw", truth.renewable_power_kw, "kW"),
            ("grid_power_kw", truth.grid_power_kw, "kW"),
            ("load_power_kw", truth.load_power_kw, "kW"),
            ("unserved_power_kw", truth.unserved_power_kw, "kW"),
            ("power_supply_ratio", truth.power_supply_ratio, "ratio"),
        )
        for metric, value, unit in energy_points:
            obs.append(Observation(
                point_id=metric, greenhouse_id=greenhouse_id,
                zone_id="GLOBAL", metric=metric, value=round(float(value), 4),
                unit=unit, observed_at=sim_time, received_at=sim_time,
                quality=Quality.GOOD, sequence=sequence, run_id=run_id,
                scenario_id=scenario_id, seed=seed,
            ))

        return obs

    # 故障注入接口（由 FaultScheduler 调用）
    def inject_stuck(self, point_id: str, value: float) -> None:
        self._stuck_points.add(point_id)
        self._stuck_values[point_id] = value

    def inject_drift(self, point_id: str, rate_per_hour: float) -> None:
        self._drift_points[point_id] = rate_per_hour

    def inject_spike(self, point_id: str) -> None:
        self._spike_points[point_id] = 1.0

    def inject_disconnect(self, point_id: str) -> None:
        self._disconnected_points.add(point_id)

    def clear_fault(self, point_id: str) -> None:
        self._stuck_points.discard(point_id)
        self._drift_points.pop(point_id, None)
        self._disconnected_points.discard(point_id)
