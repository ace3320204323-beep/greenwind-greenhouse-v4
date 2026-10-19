"""评价指标计算（任务书 14）。

在仿真过程中累积阶段指标，结束时汇总。指标分环境、控制、能源、安全四类。
能源模块第一版关闭，能源指标记为 0 且不伪造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MetricsAccumulator:
    # 环境
    temp_errors: list[float] = field(default_factory=list)
    temp_in_range: int = 0
    temp_samples: int = 0
    humidity_in_range: int = 0
    humidity_samples: int = 0
    violation_degree_minutes: float = 0.0
    zone_max_temp_diff: float = 0.0

    # 控制
    command_count: int = 0
    actuator_travel_total_pct: float = 0.0
    device_switch_count: int = 0
    missed_control_cycles: int = 0

    # 安全
    invalid_command_rejected: int = 0
    fallback_success: int = 0
    unsafe_action_count: int = 0
    sensor_missing: int = 0
    sensor_total: int = 0

    # 命令-反馈匹配（零指令时 match_ratio 应为 null）
    command_issued: int = 0
    command_feedback_match: int = 0

    # 恢复时间（参数突变后）
    shift_detected: bool = False
    shift_sim_hour: float = 0.0
    recovery_sim_hour: float | None = None

    _last_actuator: dict[str, float] = field(default_factory=dict)

    def record_step(self, truth, target_min: float, target_max: float,
                    humidity_min: float, humidity_max: float,
                    obs: list, dt_seconds: float,
                    actuators: dict[str, float]) -> None:
        """记录单步指标。"""
        # 环境（取三区均值温度）
        temps = [truth.zone_east_air_temperature_c,
                 truth.zone_center_air_temperature_c,
                 truth.zone_west_air_temperature_c]
        mean_temp = sum(temps) / 3.0
        self.temp_samples += 1
        if target_min <= mean_temp <= target_max:
            self.temp_in_range += 1
        else:
            err = max(target_min - mean_temp, mean_temp - target_max, 0.0)
            self.temp_errors.append(err)
            self.violation_degree_minutes += err * dt_seconds / 60.0

        # 三区最大温差
        self.zone_max_temp_diff = max(self.zone_max_temp_diff,
                                      max(temps) - min(temps))

        # 湿度（取中心区）
        self.humidity_samples += 1
        rh_center = None
        for o in obs:
            if o.metric == "relative_humidity_pct" and o.zone_id == "CENTER":
                rh_center = o.value
                break
        if rh_center is not None and humidity_min <= rh_center <= humidity_max:
            self.humidity_in_range += 1

        # 传感器缺失
        for o in obs:
            self.sensor_total += 1
            if o.value is None:
                self.sensor_missing += 1

        # 执行器累计行程与启停
        for k, v in actuators.items():
            prev = self._last_actuator.get(k, v)
            self.actuator_travel_total_pct += abs(v - prev)
            self._last_actuator[k] = v

    def summarize(self, heater_kwh: float = 0.0, fan_kwh: float = 0.0) -> dict[str, Any]:
        """汇总为 metrics dict。"""
        n = self.temp_samples or 1
        temp_mae = sum(self.temp_errors) / max(1, len(self.temp_errors)) if self.temp_errors else 0.0
        temp_rmse = math.sqrt(sum(e * e for e in self.temp_errors) / max(1, len(self.temp_errors))) if self.temp_errors else 0.0

        return {
            "environment": {
                "temperature_mae_c": round(temp_mae, 4),
                "temperature_rmse_c": round(temp_rmse, 4),
                "temperature_in_range_ratio": round(self.temp_in_range / n, 4),
                "temperature_violation_degree_minutes": round(self.violation_degree_minutes, 2),
                "humidity_in_range_ratio": round(self.humidity_in_range / max(1, self.humidity_samples), 4),
                "zone_temperature_max_difference_c": round(self.zone_max_temp_diff, 3),
            },
            "control": {
                "recovery_time_after_shift_min": round((self.recovery_sim_hour - self.shift_sim_hour) * 60.0, 1)
                if self.recovery_sim_hour is not None else None,
                "command_count": self.command_count,
                "actuator_travel_total_pct": round(self.actuator_travel_total_pct, 2),
                "device_switch_count": self.device_switch_count,
                "missed_control_cycles": self.missed_control_cycles,
            },
            "energy": {
                "heater_energy_kwh": round(heater_kwh, 3),
                "fan_energy_kwh": round(fan_kwh, 3),
                "total_energy_kwh": round(heater_kwh + fan_kwh, 3),
                "renewable_utilization_ratio": 0.0,
                "critical_load_loss_time_min": 0.0,
            },
            "safety": {
                "invalid_command_rejected_count": self.invalid_command_rejected,
                "fault_detection_delay_s": None,
                "fallback_success_count": self.fallback_success,
                "unsafe_action_count": self.unsafe_action_count,
                "sensor_missing_ratio": round(self.sensor_missing / max(1, self.sensor_total), 4),
                "command_feedback_match_ratio": (
                    round(self.command_feedback_match / self.command_issued, 4)
                    if self.command_issued > 0 else None),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summarize()
