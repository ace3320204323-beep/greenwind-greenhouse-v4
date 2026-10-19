"""天气模型。

生成室外温度、湿度、太阳辐射、风速、降雨，作为棚外边界条件。
- 天气连续变化，不允许每秒无规律跳变（任务书 9.2 / 18.5）。
- 支持晴冷 clear_cold、阴天高湿 overcast_humid、寒潮 cold_snap、连续阴天 overcast_series 四类场景。
- 一阶相关扰动：noise(t+1) = alpha * noise(t) + epsilon(t)。
- 所有随机量来自同一个受控随机数生成器，相同 seed 可复现。

时间基准：sim_seconds 从仿真起点（sim_start，默认 08:00）起算，
hour_of_day 必须加上起始钟点偏移，才能正确反映绝对钟点与昼夜。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import WeatherParams


@dataclass
class WeatherState:
    """一个时刻的室外天气。"""
    outdoor_air_temperature_c: float
    outdoor_relative_humidity_pct: float
    outdoor_solar_irradiance_w_m2: float
    outdoor_wind_speed_m_s: float
    outdoor_rainfall_mm_h: float
    light_par_umol_m2_s: float   # 光合有效辐射，由太阳辐射换算
    scene: str


class WeatherModel:
    """按场景时间表生成连续室外天气。"""

    # 场景基准（叠加到 base 上）
    SCENE_PROFILES = {
        "clear_cold": {
            "solar_scale": 1.0, "cloudiness": 0.0, "temp_offset": 0.0,
            "humidity_offset": -10.0, "wind_scale": 1.0, "rain": 0.0,
        },
        "overcast_humid": {
            "solar_scale": 0.3, "cloudiness": 0.7, "temp_offset": 1.0,
            "humidity_offset": 15.0, "wind_scale": 0.8, "rain": 0.0,
        },
        "cold_snap": {
            "solar_scale": 0.5, "cloudiness": 0.5, "temp_offset": -6.0,
            "humidity_offset": 0.0, "wind_scale": 1.8, "rain": 0.0,
        },
        "overcast_series": {
            "solar_scale": 0.25, "cloudiness": 0.8, "temp_offset": 2.0,
            "humidity_offset": 18.0, "wind_scale": 0.7, "rain": 0.5,
        },
    }

    def __init__(self, params: WeatherParams, seed: int, schedule: list[dict],
                 hour_offset: float = 8.0):
        self.params = params
        self.rng = random.Random(seed ^ 0x9E3779B9)  # 独立随机流
        self.schedule = schedule
        self.hour_offset = hour_offset  # 仿真起点对应的绝对钟点
        # 一阶相关扰动状态
        self._noise_temp = 0.0
        self._noise_solar = 0.0
        self._noise_wind = 0.0

    def _scene_at(self, hour: float) -> dict:
        """返回 hour 时刻生效的场景 profile，缺省用第一个或 clear_cold。"""
        current = self.SCENE_PROFILES["clear_cold"]
        name = "clear_cold"
        for seg in self.schedule:
            if seg.get("start_hour", 0.0) <= hour < seg.get("end_hour", 1e18):
                name = seg.get("scene", "clear_cold")
                current = self.SCENE_PROFILES.get(name, current)
                break
        return {"name": name, **current}

    def _correlated(self, state: float, sigma: float) -> float:
        """一阶相关扰动更新并返回新值。"""
        alpha = self.params.weather_noise_alpha
        state = alpha * state + self.rng.gauss(0.0, sigma)
        return state

    def step(self, sim_seconds: float) -> WeatherState:
        """sim_seconds 为当前仿真时刻（从起点起算的秒）。"""
        p = self.params
        # 绝对钟点 = 起始偏移 + 已流逝时间
        hour_of_day = ((self.hour_offset * 3600.0 + sim_seconds) % 86400.0) / 3600.0
        day = sim_seconds / 86400.0

        scene = self._scene_at(sim_seconds / 3600.0)

        # 太阳辐射：白天正弦，受云量抑制
        daylight = max(0.0, math.sin(math.pi * (hour_of_day - 6.5) / 11.0))
        solar = p.base_solar_irradiance_w_m2 * daylight * scene["solar_scale"]
        # 云量引入部分随机遮挡（连续相关）。夜间太阳辐射必须为零，
        # 不能让相关噪声制造“午夜太阳”并触发白天目标温区。
        self._noise_solar = self._correlated(self._noise_solar, p.weather_noise_sigma_solar)
        if daylight <= 0.0:
            solar = 0.0
        else:
            solar = max(0.0, solar + self._noise_solar * (0.5 + scene["cloudiness"]))

        # 室外温度：日正弦 + 场景偏移 + 逐日缓慢漂移 + 相关噪声
        self._noise_temp = self._correlated(self._noise_temp, p.weather_noise_sigma_temp)
        daily = p.daily_temp_amplitude_c * math.sin(2 * math.pi * (hour_of_day - 8.0) / 24.0)
        slow_drift = 3.0 * math.sin(2 * math.pi * day / 6.0)  # 多日尺度波动
        temp = (p.base_outdoor_temp_c + scene["temp_offset"] + daily + slow_drift
                + self._noise_temp)

        # 湿度：白天低、夜间高，叠加场景偏移与相关噪声
        humidity = (p.base_humidity_pct + scene["humidity_offset"]
                    - 15.0 * daylight + self._noise_temp * 1.5)
        humidity = max(15.0, min(100.0, humidity))

        # 风速：基础 + 场景缩放 + 日变化 + 相关噪声
        self._noise_wind = self._correlated(self._noise_wind, p.weather_noise_sigma_temp * 0.5)
        wind = max(0.0, p.base_wind_speed_m_s * scene["wind_scale"]
                   + 0.8 * abs(math.sin(2 * math.pi * hour_of_day / 8.0))
                   + self._noise_wind)

        rain = scene["rain"]

        # PAR 换算（太阳辐射 W/m² → μmol/(m²·s)，近似系数 0.45）
        par = solar * 0.45

        return WeatherState(
            outdoor_air_temperature_c=round(temp, 3),
            outdoor_relative_humidity_pct=round(humidity, 3),
            outdoor_solar_irradiance_w_m2=round(solar, 3),
            outdoor_wind_speed_m_s=round(wind, 3),
            outdoor_rainfall_mm_h=round(rain, 3),
            light_par_umol_m2_s=round(par, 3),
            scene=scene["name"],
        )
