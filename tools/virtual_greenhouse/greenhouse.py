"""棚内物理模型（GreenhousePlant）。

维护并更新棚内真实物理状态。这是"被控对象"，与控制器内部的简化一阶灰箱模型
**不同构**：三区空气、冠层、基质、棚架蓄热体分别持有独立温度状态，通过
相邻区交换、冠层对流、基质蓄热、棚架蓄放热、通风换热与棚体散热耦合。

每个温度节点都按能量平衡积分：C_i * dT_i/dt = sum(Q_in) - sum(Q_out)。
因此加热、停热或室外温度突变后，棚温不会瞬时跳变，而是呈现可配置、
可验证的升温与衰减过程。

物理量标定（P0.1-physical-calibrated）：
- 通风系数 true_ventilation_coefficient = 3.0（风速 2m/s 时换气约 9.6 次/h）
- 棚体基础传热 true_heat_transfer_coefficient = 5.0 W/(m²·K)（即 U_cover）
- 保温帘修正系数独立：每 10% 关闭比例使 U 降 4%，全关时 U_eff ≈ 3.0
- 冠层热容 200 kJ/K 强对流耦合；基质热容 1500 kJ/K 白天蓄热夜间释放
- 加热 30kW；加热效率 heater_efficiency 可被"加热能力下降"突变调低

湿度质量平衡（P1 诊断后修复）：
- 绝对湿度参与质量平衡，蒸腾/基质蒸发/灌溉增湿，通风排湿
- 结露：绝对湿度超过饱和值时，超出部分按强约束凝结，RH 物理封顶 100%
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import GreenhouseConfig, ZoneParams, PlantParams
from .schemas import TrueState, ZoneId


ZONES = [ZoneId.EAST, ZoneId.CENTER, ZoneId.WEST]

NEIGHBORS = {
    ZoneId.EAST: [ZoneId.CENTER],
    ZoneId.CENTER: [ZoneId.EAST, ZoneId.WEST],
    ZoneId.WEST: [ZoneId.CENTER],
}


@dataclass
class PlantInternal:
    air_temp: dict[str, float] = field(default_factory=lambda: {
        "EAST": 22.0, "CENTER": 22.0, "WEST": 22.0,
    })
    air_abs_humidity: dict[str, float] = field(default_factory=lambda: {
        "EAST": 8.0, "CENTER": 8.0, "WEST": 8.0,
    })
    canopy_temp: float = 21.0
    substrate_temp: float = 20.0
    structure_temp: float = 19.0
    cover_temp: float = 18.0
    co2_ppm: float = 450.0
    substrate_moisture: float = 60.0
    canopy_transpiration: float = 0.0
    condensation_rate: float = 0.0


class GreenhousePlant:
    HEATER_RATED_KW = 30.0

    def __init__(self, config: GreenhouseConfig):
        self.cfg = config
        self.plant: PlantParams = config.plant
        self.zones: dict[str, ZoneParams] = config.zones
        self.state = PlantInternal()

        self.state.air_temp["EAST"] = 21.8
        self.state.air_temp["CENTER"] = 22.2
        self.state.air_temp["WEST"] = 21.5

        self.true_params = {
            "true_heat_transfer_coefficient": 5.0,
            "true_ventilation_coefficient": 3.0,
            "true_solar_gain_coefficient": 0.03,
            "true_transpiration_coefficient": 0.02,
            "true_air_leakage_rate": 0.003,
        }
        self.curtain_u_reduction_per_10pct = 0.04
        self.heater_efficiency = 1.0

        # 所有热惯性参数只来自场景配置，避免“配置一套、代码硬编码另一套”。
        self.canopy_heat_capacity_kj_k = self.plant.canopy_heat_capacity_kj_k
        self.canopy_air_coupling_kw_k = self.plant.canopy_air_exchange_kw_k
        self.substrate_heat_capacity_kj_k = self.plant.substrate_heat_capacity_kj_k
        self.substrate_air_coupling_kw_k = self.plant.substrate_air_exchange_kw_k
        self.structure_heat_capacity_kj_k = self.plant.structure_heat_capacity_kj_k
        self.structure_air_coupling_kw_k = self.plant.structure_air_exchange_kw_k

        if min(self.canopy_heat_capacity_kj_k, self.substrate_heat_capacity_kj_k,
               self.structure_heat_capacity_kj_k) <= 0:
            raise ValueError("热容必须大于 0 kJ/K")

    def _zone_heat_capacity(self, zid: ZoneId) -> float:
        """返回分区有效热容 (kJ/K)。

        air_heat_capacity_kj_k 由棚体体积与空气物性给出，thermal_inertia
        表示该分区与附近设施、管路等的等效耦合差异。
        """
        zp = self.zones[zid]
        capacity = zp.air_heat_capacity_kj_k * zp.thermal_inertia
        if capacity <= 0:
            raise ValueError(f"{zid.value} 分区有效热容必须大于 0 kJ/K")
        return capacity

    # ------------------------------------------------------------------
    def _cover_u_effective(self, curtain_pct, wind_speed):
        u_cover = self.true_params["true_heat_transfer_coefficient"]
        curtain_factor = 1.0 - self.curtain_u_reduction_per_10pct * (curtain_pct / 10.0)
        curtain_factor = max(0.4, min(1.0, curtain_factor))
        wind_factor = 1.0 + 0.03 * wind_speed
        return u_cover * curtain_factor * wind_factor

    def _ventilation_flow(self, vent_pct, fan_pct, wind_speed, faulted):
        if faulted:
            return 0.0
        base = self.true_params["true_ventilation_coefficient"]
        flow = (base * (vent_pct / 100.0) * (1.0 + 0.3 * wind_speed)
                + base * 0.5 * (fan_pct / 100.0)
                + self.true_params["true_air_leakage_rate"])
        return flow

    # ------------------------------------------------------------------
    def step(self, sim_seconds, dt_seconds, weather, actuators, rng):
        p = self.plant
        zones = self.zones
        st = self.state

        vent = actuators.get("vent_01", 0.0)
        fan = actuators.get("fan_01", 0.0)
        curtain = actuators.get("curtain_01", 0.0)
        heater = actuators.get("heater_01", 0.0)
        irrigation = actuators.get("irrigation_01", 0.0)

        solar_in = weather.outdoor_solar_irradiance_w_m2 * (1.0 - 0.5 * curtain / 100.0)
        u_eff = self._cover_u_effective(curtain, weather.outdoor_wind_speed_m_s)
        vent_flow = self._ventilation_flow(vent, fan, weather.outdoor_wind_speed_m_s, False)
        t_out = weather.outdoor_air_temperature_c

        lai = p.leaf_area_index
        transpiration = (self.true_params["true_transpiration_coefficient"]
                         * lai * max(0.0, solar_in) / 850.0)
        st.canopy_transpiration = transpiration

        dt = dt_seconds

        mean_air = sum(st.air_temp.values()) / 3.0

        new_air_temp: dict[str, float] = {}
        for zid in ZONES:
            zp = zones[zid]
            t = st.air_temp[zid]
            c_zone = self._zone_heat_capacity(zid)

            q_solar_kw = (self.true_params["true_solar_gain_coefficient"]
                          * solar_in * p.cover_area_m2 / 1000.0) * zp.thermal_inertia / 3.0
            q_heater_kw = heater / 100.0 * self.HEATER_RATED_KW * self.heater_efficiency / 3.0
            q_cover_kw = u_eff * p.cover_area_m2 * (t - t_out) / 1000.0 / 3.0
            q_vent_kw = (p.air_density_kg_m3 * p.air_heat_capacity_j_kg_k
                         * vent_flow * zp.vent_exposure * (t - t_out) / 1000.0)
            q_canopy_kw = self.canopy_air_coupling_kw_k * (st.canopy_temp - t) / 3.0
            q_substrate_kw = self.substrate_air_coupling_kw_k * (st.substrate_temp - t) / 3.0
            q_structure_kw = self.structure_air_coupling_kw_k * (st.structure_temp - t) / 3.0

            q_neighbor_kw = 0.0
            for nb in NEIGHBORS[zid]:
                q_neighbor_kw += p.zone_neighbor_exchange_kw_k * (st.air_temp[nb] - t)

            q_net_kw = (q_solar_kw + q_heater_kw - q_cover_kw - q_vent_kw
                        + q_canopy_kw + q_substrate_kw + q_structure_kw
                        + q_neighbor_kw)

            dtemp = q_net_kw * dt / c_zone
            new_air_temp[zid] = t + dtemp

        st.air_temp = new_air_temp

        # 冠层温度
        canopy_absorbed_kw = (self.true_params["true_solar_gain_coefficient"]
                              * solar_in * p.cover_area_m2 / 1000.0) * 0.15
        canopy_q_kw = (canopy_absorbed_kw
                       + self.canopy_air_coupling_kw_k * (mean_air - st.canopy_temp)
                       - transpiration * 2.45 / 1000.0)
        st.canopy_temp += canopy_q_kw * dt / self.canopy_heat_capacity_kj_k

        # 基质温度
        substrate_q_kw = self.substrate_air_coupling_kw_k * (mean_air - st.substrate_temp)
        st.substrate_temp += substrate_q_kw * dt / self.substrate_heat_capacity_kj_k

        # 棚架、管路和棚内设施等效蓄热体。它不是额外热源，而是与空气
        # 双向换热：升温时吸收热量，停热后再向空气释放，形成可观测的热惯性。
        structure_q_kw = self.structure_air_coupling_kw_k * (mean_air - st.structure_temp)
        st.structure_temp += structure_q_kw * dt / self.structure_heat_capacity_kj_k

        # 棚体温度
        st.cover_temp += ((0.01 * (mean_air - st.cover_temp)
                           + 0.005 * (t_out - st.cover_temp)) * dt / 60.0)

        # 湿度质量平衡（显式质量守恒：超饱和水汽转化为结露并从空气扣除）
        total_cond = 0.0
        for zid in ZONES:
            zp = zones[zid]
            w = st.air_abs_humidity[zid]
            e_crop = transpiration / 3.0
            e_substrate = p.substrate_evaporation_base * (solar_in / 850.0) / 3.0
            e_irrigation = irrigation * 0.2 / 60.0
            t = st.air_temp[zid]
            outside_abs_hum = self._rh_to_abs(weather.outdoor_relative_humidity_pct, t_out)
            e_vent = vent_flow * zp.vent_exposure * (w - outside_abs_hum) * 0.01

            # 结露：先把"超过饱和值"的水汽显式凝结（质量守恒，非截断隐藏）
            sat_now = self._saturation_abs_humidity(t)
            # 速率型结露：每步最多凝结一定比例，避免一步全凝结导致震荡
            excess = max(0.0, w - sat_now)
            e_cond_rate = excess * 0.5  # 半衰期式凝结（一步凝结剩余过量的一半）

            # 质量平衡（增湿 - 通风排湿 - 结露），dt/100 为时间缩放
            dw = (e_crop + e_substrate + e_irrigation - e_vent - e_cond_rate) * dt / 100.0
            new_w = w + dw
            # 物理下限：绝对湿度不小于一个极小值（避免数值负）
            new_w = max(0.1, new_w)

            # 若温度骤降导致新饱和值远低于当前绝对湿度，剩余过量仍按半衰式逐步凝结，
            # 这一步已在 e_cond_rate 中体现；此处不再做硬截断，只记录结露量。
            # 记录本区结露速率（g/m³/s，供质量守恒核对）
            zone_cond = e_cond_rate * dt / 100.0
            st.air_abs_humidity[zid] = new_w
            total_cond += zone_cond

        st.condensation_rate = total_cond

        # 结露释放潜热到空气（凝结 1g 水约释放 2450 J 潜热）
        # 结露量 g/m³ · 体积 m³ = g，潜热 kJ = g · 2.45
        cond_latent_kw = total_cond * p.volume_m3 * 2.45 / 1000.0 / dt
        for zid in ZONES:
            st.air_temp[zid] += cond_latent_kw * dt / self._zone_heat_capacity(zid)

        # 基质水分
        st.substrate_moisture = max(0.0, st.substrate_moisture
                                    - (0.006 + 0.018 * solar_in / 850.0) * dt / 60.0
                                    + irrigation * dt / 3600.0)

        # CO2
        target_co2 = 420.0 if vent_flow > 0.3 else 760.0
        st.co2_ppm += (target_co2 - st.co2_ppm) * 0.01 * dt / 60.0

    # ------------------------------------------------------------------
    @staticmethod
    def _saturation_abs_humidity(temp_c):
        es = 6.112 * math.exp(17.62 * temp_c / (243.12 + temp_c))
        return 216.7 * es / (273.15 + temp_c)

    def _rh_to_abs(self, rh_pct, temp_c):
        return self._saturation_abs_humidity(temp_c) * rh_pct / 100.0

    def rh_at(self, abs_humidity, temp_c):
        sat = self._saturation_abs_humidity(temp_c)
        if sat <= 0:
            return 100.0
        return min(100.0, max(0.0, abs_humidity / sat * 100.0))

    # ------------------------------------------------------------------
    def snapshot(self, sim_time, sim_step, dt_seconds, actuators) -> TrueState:
        ts = TrueState()
        ts.sim_time = sim_time
        ts.sim_step = sim_step
        ts.dt_seconds = dt_seconds
        ts.zone_east_air_temperature_c = round(self.state.air_temp["EAST"], 3)
        ts.zone_center_air_temperature_c = round(self.state.air_temp["CENTER"], 3)
        ts.zone_west_air_temperature_c = round(self.state.air_temp["WEST"], 3)
        ts.zone_east_absolute_humidity_g_m3 = round(self.state.air_abs_humidity["EAST"], 3)
        ts.zone_center_absolute_humidity_g_m3 = round(self.state.air_abs_humidity["CENTER"], 3)
        ts.zone_west_absolute_humidity_g_m3 = round(self.state.air_abs_humidity["WEST"], 3)
        ts.canopy_temperature_c = round(self.state.canopy_temp, 3)
        ts.canopy_transpiration_g_s = round(self.state.canopy_transpiration, 4)
        ts.condensation_rate_g_m3_s = round(self.state.condensation_rate, 4)
        ts.substrate_temperature_c = round(self.state.substrate_temp, 3)
        ts.structure_temperature_c = round(self.state.structure_temp, 3)
        ts.substrate_moisture_pct = round(self.state.substrate_moisture, 3)
        ts.cover_temperature_c = round(self.state.cover_temp, 3)
        ts.indoor_co2_ppm = round(self.state.co2_ppm, 2)

        ts.actual_vent_position_pct = round(actuators.get("vent_01", 0.0), 3)
        ts.actual_fan_power_pct = round(actuators.get("fan_01", 0.0), 3)
        ts.actual_curtain_position_pct = round(actuators.get("curtain_01", 0.0), 3)
        ts.actual_heater_power_pct = round(actuators.get("heater_01", 0.0), 3)
        ts.actual_irrigation_flow_l_min = round(actuators.get("irrigation_01", 0.0), 3)

        ts.true_heat_transfer_coefficient = self.true_params["true_heat_transfer_coefficient"]
        ts.true_ventilation_coefficient = self.true_params["true_ventilation_coefficient"]
        ts.true_solar_gain_coefficient = self.true_params["true_solar_gain_coefficient"]
        ts.true_transpiration_coefficient = self.true_params["true_transpiration_coefficient"]
        ts.true_air_leakage_rate = self.true_params["true_air_leakage_rate"]
        return ts

    def apply_parameter_shift(self, parameter, factor):
        if parameter in self.true_params:
            self.true_params[parameter] *= factor

    def set_heater_efficiency(self, efficiency):
        self.heater_efficiency = max(0.0, min(1.0, efficiency))
