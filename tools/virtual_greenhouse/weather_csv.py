"""真实天气 CSV 数据接入（任务书第十七章 P1 建议项 / 收档项）。

从 CSV 读取真实气象站逐小时数据（时间、室外温度、湿度、太阳辐射、风速），
支持：
- 单位转换（温度 ℃ 直接使用；太阳辐射支持 W/m²；风速 m/s）
- 缺失值处理（NaN → 线性插值或前值填充，并标记）
- 异常值标记（超出物理范围的值标为 SUSPECT，并替换为插值）
- 保留数据来源、地点、时间范围元数据
- 按仿真时刻返回对应天气（线性插值），接口与 WeatherModel.step 对齐

CSV 格式（表头）：
    timestamp, outdoor_air_temperature_c, outdoor_relative_humidity_pct,
    outdoor_solar_irradiance_w_m2, outdoor_wind_speed_m_s
其中 timestamp 为 ISO 8601 或 'YYYY-MM-DD HH:MM'。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .weather import WeatherState


CN_TZ = timezone(timedelta(hours=8))


@dataclass
class WeatherCSVMeta:
    """真实天气数据的来源元信息。"""
    source: str = ""
    location: str = ""
    time_start: str = ""
    time_end: str = ""
    n_records: int = 0
    n_missing_filled: int = 0
    n_outliers_marked: int = 0


def _parse_time(s: str) -> float:
    """解析时间戳为 UTC epoch 秒。支持 ISO 8601 和 'YYYY-MM-DD HH:MM'。"""
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"无法解析时间戳: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CN_TZ)
    return dt.timestamp()


class WeatherCSVModel:
    """从 CSV 读取真实天气，按仿真时刻插值返回 WeatherState。"""

    def __init__(self, csv_path: str | Path, seed: int = 0,
                 meta: dict | None = None):
        self.csv_path = Path(csv_path)
        self.seed = seed
        self.meta = WeatherCSVMeta()
        if meta:
            self.meta.source = meta.get("source", "")
            self.meta.location = meta.get("location", "")
        self._times: list[float] = []
        self._temp: list[float] = []
        self._humidity: list[float] = []
        self._solar: list[float] = []
        self._wind: list[float] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        rows = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        self.meta.n_records = len(rows)
        if rows:
            self.meta.time_start = rows[0]["timestamp"]
            self.meta.time_end = rows[-1]["timestamp"]

        for r in rows:
            t = _parse_time(r["timestamp"])
            self._times.append(t)
            self._temp.append(self._num(r.get("outdoor_air_temperature_c")))
            self._humidity.append(self._num(r.get("outdoor_relative_humidity_pct")))
            self._solar.append(self._num(r.get("outdoor_solar_irradiance_w_m2")))
            self._wind.append(self._num(r.get("outdoor_wind_speed_m_s")))

        self._fill_missing()
        self._mark_outliers()

    @staticmethod
    def _num(v) -> float | None:
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _fill_missing(self) -> None:
        """缺失值处理：线性插值，端点用前后值填充。"""
        for arr in (self._temp, self._humidity, self._solar, self._wind):
            for i in range(len(arr)):
                if arr[i] is None:
                    self.meta.n_missing_filled += 1
            # 线性插值
            filled = self._interpolate(arr)
            arr[:] = filled

    @staticmethod
    def _interpolate(arr: list) -> list:
        """对含 None 的序列做线性插值，端点取最近有效值。"""
        n = len(arr)
        out = list(arr)
        # 前向填充端点
        last_valid = None
        for i in range(n):
            if out[i] is None:
                out[i] = last_valid if last_valid is not None else None
            else:
                last_valid = out[i]
        # 后向填充开头
        last_valid = None
        for i in range(n - 1, -1, -1):
            if out[i] is None:
                out[i] = last_valid
            else:
                last_valid = out[i]
        return out

    def _mark_outliers(self) -> None:
        """异常值标记：超出物理范围的值标 SUSPECT 并替换为插值。"""
        bounds = {
            "temp": (-50.0, 50.0),
            "humidity": (0.0, 100.0),
            "solar": (0.0, 1400.0),
            "wind": (0.0, 60.0),
        }
        for name, arr in zip(("temp", "humidity", "solar", "wind"),
                             (self._temp, self._humidity, self._solar, self._wind)):
            lo, hi = bounds[name]
            for i in range(len(arr)):
                if arr[i] is None or not (lo <= arr[i] <= hi):
                    self.meta.n_outliers_marked += 1
                    # 用相邻有效值替换
                    arr[i] = self._neighbor_mean(arr, i, lo, hi)

    @staticmethod
    def _neighbor_mean(arr: list, idx: int, lo: float, hi: float) -> float:
        """取异常点前后最近的有效值均值，无则取边界内默认。"""
        vals = []
        for j in range(idx - 1, -1, -1):
            if arr[j] is not None and lo <= arr[j] <= hi:
                vals.append(arr[j])
                break
        for j in range(idx + 1, len(arr)):
            if arr[j] is not None and lo <= arr[j] <= hi:
                vals.append(arr[j])
                break
        if vals:
            return sum(vals) / len(vals)
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    def step(self, sim_seconds: float) -> WeatherState:
        """按仿真时刻返回真实天气（线性插值）。

        sim_seconds 是仿真内部时刻（从仿真起点起算）。真实天气按
        第一个时间戳对齐：sim_seconds=0 对应 CSV 第一条记录。
        """
        if not self._times:
            return WeatherState(0.0, 70.0, 0.0, 0.0, 0.0, 0.0, "csv_empty")

        t0 = self._times[0]
        # 仿真时刻映射到真实时间轴：sim_seconds 秒 = 从 t0 起算
        target = t0 + sim_seconds
        t_end = self._times[-1]
        if target >= t_end:
            # 超出数据范围：保持最后一条
            i = len(self._times) - 1
            temp, hum, sol, wind = (self._temp[i], self._humidity[i],
                                    self._solar[i], self._wind[i])
        else:
            # 二分查找插值区间
            lo, hi = 0, len(self._times) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if self._times[mid] <= target:
                    lo = mid
                else:
                    hi = mid
            # 线性插值
            t_lo, t_hi = self._times[lo], self._times[hi]
            frac = 0.0 if t_hi == t_lo else (target - t_lo) / (t_hi - t_lo)
            temp = self._temp[lo] + frac * (self._temp[hi] - self._temp[lo])
            hum = self._humidity[lo] + frac * (self._humidity[hi] - self._humidity[lo])
            sol = self._solar[lo] + frac * (self._solar[hi] - self._solar[lo])
            wind = self._wind[lo] + frac * (self._wind[hi] - self._wind[lo])

        # 太阳辐射非负，夜间为 0
        sol = max(0.0, sol)
        # PAR 换算
        par = sol * 0.45

        return WeatherState(
            outdoor_air_temperature_c=round(temp, 3),
            outdoor_relative_humidity_pct=round(max(0.0, min(100.0, hum)), 3),
            outdoor_solar_irradiance_w_m2=round(sol, 3),
            outdoor_wind_speed_m_s=round(max(0.0, wind), 3),
            outdoor_rainfall_mm_h=0.0,
            light_par_umol_m2_s=round(par, 3),
            scene="real_weather_csv",
        )

    def summary(self) -> dict:
        return {
            "source": self.meta.source,
            "location": self.meta.location,
            "time_start": self.meta.time_start,
            "time_end": self.meta.time_end,
            "n_records": self.meta.n_records,
            "n_missing_filled": self.meta.n_missing_filled,
            "n_outliers_marked": self.meta.n_outliers_marked,
        }
