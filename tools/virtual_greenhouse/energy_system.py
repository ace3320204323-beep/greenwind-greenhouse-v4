"""单棚清洁能源与储能子系统。

模块只处理功率平衡、SOC、市电状态与供电缺口，不直接修改环境温度。
虚拟大棚根据本模块返回的 effective_actuators 计算实际热/通风效果，从而形成
“设备指令→用电负荷→市电/新能源/储能供电→实际设备效果”闭环。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import DeviceSpec, EnergyParams


VALID_GRID_STATUS = {"NORMAL", "LIMITED", "OUTAGE"}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class EnergyState:
    battery_soc_pct: float
    battery_power_kw: float = 0.0
    pv_power_kw: float = 0.0
    wind_power_kw: float = 0.0
    renewable_power_kw: float = 0.0
    grid_power_kw: float = 0.0
    load_power_kw: float = 0.0
    served_power_kw: float = 0.0
    unserved_power_kw: float = 0.0
    renewable_curtailed_kw: float = 0.0
    power_supply_ratio: float = 1.0
    grid_status: str = "NORMAL"
    cumulative_grid_kwh: float = 0.0
    cumulative_renewable_used_kwh: float = 0.0
    cumulative_unserved_kwh: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyStepResult:
    state: EnergyState
    requested_actuators: dict[str, float]
    effective_actuators: dict[str, float]
    device_supply_ratio: float
    balance_residual_kw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "requested_actuators": dict(self.requested_actuators),
            "effective_actuators": dict(self.effective_actuators),
            "device_supply_ratio": self.device_supply_ratio,
            "balance_residual_kw": self.balance_residual_kw,
        }


class EnergySystem:
    """可独立测试的功率平衡与储能调度模型。"""

    def __init__(self, params: EnergyParams, devices: list[DeviceSpec],
                 enabled: bool = True, grid_schedule: list[dict[str, Any]] | None = None):
        self.params = params
        self.enabled = bool(enabled)
        self.device_specs = {device.device_id: device for device in devices}
        self.grid_schedule = list(grid_schedule or [])
        self.grid_status_override: str | None = None
        self.state = EnergyState(
            battery_soc_pct=_clip(
                params.initial_soc_pct, params.min_soc_pct, params.max_soc_pct
            )
        )

    def set_grid_status_override(self, status: str | None) -> None:
        if status is None or str(status).upper() == "AUTO":
            self.grid_status_override = None
            return
        normalized = str(status).upper()
        if normalized not in VALID_GRID_STATUS:
            raise ValueError(f"未知市电状态：{status}")
        self.grid_status_override = normalized

    def _grid_status(self, sim_hour: float) -> str:
        if self.grid_status_override is not None:
            return self.grid_status_override
        for segment in self.grid_schedule:
            start = float(segment.get("start_hour", 0.0))
            end = float(segment.get("end_hour", 0.0))
            if start <= sim_hour < end:
                status = str(segment.get("status", "NORMAL")).upper()
                return status if status in VALID_GRID_STATUS else "NORMAL"
        return "NORMAL"

    def _renewable_power(self, solar_w_m2: float, wind_m_s: float) -> tuple[float, float]:
        p = self.params
        pv = p.pv_peak_power_kw * _clip(solar_w_m2 / 1000.0, 0.0, 1.2)

        wind_speed = max(0.0, float(wind_m_s))
        if wind_speed < p.wind_cut_in_m_s or wind_speed >= p.wind_cut_out_m_s:
            wind = 0.0
        elif wind_speed >= p.wind_rated_speed_m_s:
            wind = p.wind_rated_power_kw
        else:
            span = max(0.1, p.wind_rated_speed_m_s - p.wind_cut_in_m_s)
            fraction = (wind_speed - p.wind_cut_in_m_s) / span
            wind = p.wind_rated_power_kw * fraction ** 3
        efficiency = _clip(p.inverter_efficiency, 0.01, 1.0)
        return pv * efficiency, wind * efficiency

    def _device_load(self, actuators: Mapping[str, float], actuator_states=None) -> tuple[float, dict[str, float]]:
        loads: dict[str, float] = {}
        for device_id, spec in self.device_specs.items():
            actual = max(0.0, float(actuators.get(device_id, 0.0)))
            if spec.kind in {"VENT", "CURTAIN"}:
                moving = False
                if actuator_states is not None and device_id in actuator_states:
                    state = actuator_states[device_id]
                    moving = abs(float(state.target) - float(state.actual)) > spec.deadband
                loads[device_id] = spec.rated_power_kw if moving else 0.0
            else:
                loads[device_id] = spec.rated_power_kw * _clip(actual / 100.0, 0.0, 1.0)
        return sum(loads.values()), loads

    def _available_discharge_kw(self, dt_hours: float) -> float:
        p = self.params
        usable_kwh = max(
            0.0,
            (self.state.battery_soc_pct - p.min_soc_pct) / 100.0
            * p.battery_capacity_kwh,
        )
        by_energy = usable_kwh * _clip(p.discharge_efficiency, 0.01, 1.0) / max(dt_hours, 1e-9)
        return min(p.max_discharge_power_kw, by_energy)

    def _available_charge_kw(self, dt_hours: float) -> float:
        p = self.params
        room_kwh = max(
            0.0,
            (p.max_soc_pct - self.state.battery_soc_pct) / 100.0
            * p.battery_capacity_kwh,
        )
        by_energy = room_kwh / max(_clip(p.charge_efficiency, 0.01, 1.0) * dt_hours, 1e-9)
        return min(p.max_charge_power_kw, by_energy)

    def step(self, sim_seconds: float, dt_seconds: float, weather,
             actuators: Mapping[str, float], actuator_states=None) -> EnergyStepResult:
        requested = {key: float(value) for key, value in actuators.items()}
        if not self.enabled:
            self.state = EnergyState(
                battery_soc_pct=self.state.battery_soc_pct,
                load_power_kw=0.0,
                served_power_kw=0.0,
                grid_status="DISABLED",
            )
            return EnergyStepResult(self.state, requested, requested, 1.0, 0.0)

        p = self.params
        dt_hours = max(float(dt_seconds) / 3600.0, 1e-9)
        sim_hour = float(sim_seconds) / 3600.0
        grid_status = self._grid_status(sim_hour)
        pv_kw, wind_kw = self._renewable_power(
            weather.outdoor_solar_irradiance_w_m2,
            weather.outdoor_wind_speed_m_s,
        )
        renewable_kw = pv_kw + wind_kw
        device_load_kw, _ = self._device_load(requested, actuator_states)
        base_kw = max(0.0, p.base_critical_load_kw)
        total_load_kw = base_kw + device_load_kw

        renewable_to_load = min(renewable_kw, total_load_kw)
        remaining_kw = total_load_kw - renewable_to_load
        grid_cap = {
            "NORMAL": p.grid_max_import_kw,
            "LIMITED": p.grid_limited_import_kw,
            "OUTAGE": 0.0,
        }[grid_status]

        grid_kw = 0.0
        discharge_kw = 0.0
        if grid_status == "NORMAL":
            grid_kw = min(remaining_kw, max(0.0, p.peak_shaving_threshold_kw), grid_cap)
            remaining_kw -= grid_kw
            discharge_kw = min(remaining_kw, self._available_discharge_kw(dt_hours))
            remaining_kw -= discharge_kw
            extra_grid = min(remaining_kw, max(0.0, grid_cap - grid_kw))
            grid_kw += extra_grid
            remaining_kw -= extra_grid
        else:
            grid_kw = min(remaining_kw, max(0.0, grid_cap))
            remaining_kw -= grid_kw
            discharge_kw = min(remaining_kw, self._available_discharge_kw(dt_hours))
            remaining_kw -= discharge_kw

        unserved_kw = max(0.0, remaining_kw)
        served_kw = total_load_kw - unserved_kw

        surplus_renewable_kw = max(0.0, renewable_kw - renewable_to_load)
        charge_kw = min(surplus_renewable_kw, self._available_charge_kw(dt_hours))
        curtailed_kw = max(0.0, surplus_renewable_kw - charge_kw)

        charge_eff = _clip(p.charge_efficiency, 0.01, 1.0)
        discharge_eff = _clip(p.discharge_efficiency, 0.01, 1.0)
        energy_change_kwh = charge_kw * charge_eff * dt_hours - discharge_kw / discharge_eff * dt_hours
        stored_kwh = self.state.battery_soc_pct / 100.0 * p.battery_capacity_kwh
        stored_kwh = _clip(
            stored_kwh + energy_change_kwh,
            p.min_soc_pct / 100.0 * p.battery_capacity_kwh,
            p.max_soc_pct / 100.0 * p.battery_capacity_kwh,
        )
        soc_pct = stored_kwh / max(p.battery_capacity_kwh, 1e-9) * 100.0

        device_served_kw = max(0.0, served_kw - min(base_kw, served_kw))
        supply_ratio = 1.0 if device_load_kw <= 1e-9 else _clip(device_served_kw / device_load_kw, 0.0, 1.0)
        effective = dict(requested)
        for device_id in ("heater_01", "fan_01", "irrigation_01"):
            if device_id in effective:
                effective[device_id] *= supply_ratio

        renewable_used_kw = renewable_to_load + charge_kw
        previous = self.state
        self.state = EnergyState(
            battery_soc_pct=soc_pct,
            battery_power_kw=discharge_kw - charge_kw,
            pv_power_kw=pv_kw,
            wind_power_kw=wind_kw,
            renewable_power_kw=renewable_kw,
            grid_power_kw=grid_kw,
            load_power_kw=total_load_kw,
            served_power_kw=served_kw,
            unserved_power_kw=unserved_kw,
            renewable_curtailed_kw=curtailed_kw,
            power_supply_ratio=0.0 if total_load_kw <= 0 else served_kw / total_load_kw,
            grid_status=grid_status,
            cumulative_grid_kwh=previous.cumulative_grid_kwh + grid_kw * dt_hours,
            cumulative_renewable_used_kwh=(
                previous.cumulative_renewable_used_kwh + renewable_used_kw * dt_hours
            ),
            cumulative_unserved_kwh=previous.cumulative_unserved_kwh + unserved_kw * dt_hours,
        )
        # 供给残差：新能源+市电+放电-充电-有效负荷-弃电。
        residual = renewable_kw + grid_kw + discharge_kw - charge_kw - served_kw - curtailed_kw
        return EnergyStepResult(self.state, requested, effective, supply_ratio, residual)

    def apply_to_truth(self, truth) -> None:
        s = self.state
        truth.battery_soc_pct = round(s.battery_soc_pct, 4)
        truth.battery_power_kw = round(s.battery_power_kw, 4)
        truth.pv_power_kw = round(s.pv_power_kw, 4)
        truth.wind_power_kw = round(s.wind_power_kw, 4)
        truth.renewable_power_kw = round(s.renewable_power_kw, 4)
        truth.grid_power_kw = round(s.grid_power_kw, 4)
        truth.load_power_kw = round(s.load_power_kw, 4)
        truth.unserved_power_kw = round(s.unserved_power_kw, 4)
        truth.renewable_curtailed_kw = round(s.renewable_curtailed_kw, 4)
        truth.power_supply_ratio = round(s.power_supply_ratio, 6)
        truth.grid_status = s.grid_status

    def summary(self) -> dict[str, Any]:
        s = self.state
        return {
            "battery_soc_pct": round(s.battery_soc_pct, 3),
            "cumulative_grid_kwh": round(s.cumulative_grid_kwh, 4),
            "cumulative_renewable_used_kwh": round(s.cumulative_renewable_used_kwh, 4),
            "cumulative_unserved_kwh": round(s.cumulative_unserved_kwh, 4),
            "grid_status": s.grid_status,
        }
