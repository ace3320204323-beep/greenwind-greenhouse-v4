"""绿风润耕单棚 SAC 训练环境 v0.2（安全与能耗奖励整改版）。

把独立虚拟大棚封装为 Gymnasium 环境，并将 Transformer 的 10/30 分钟
预测结果作为状态特征。第一版只负责形成可训练的闭环，不负责启动训练。

动作（均由 SAC 输出 [-1, 1]，再映射到 0~100%）：
    0 heater   加热功率
    1 vent     通风开度
    2 curtain  卷帘/遮阳帘开度
    3 fan      风机功率

控制周期默认 5 分钟；虚拟大棚仍按 1 分钟物理步长推进。
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generate_data_v2 import (
    BASE_SCENARIO,
    commands_from_targets,
    decide_targets,
    enforce_sensor_bounds,
    feedback_actuals,
    observation_map,
    obs_value,
    quality_of,
    sequential_schedule,
)
from transformer_inference import TransformerPredictor
from virtual_greenhouse.config import load_config
from virtual_greenhouse.runner import VirtualGreenhouse


AI_PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = AI_PIPELINE_DIR / "outputs_v2_1_residual" / "transformer_v2.pt"
DEFAULT_META_PATH = AI_PIPELINE_DIR / "outputs_v2_1_residual" / "transformer_v2_meta.json"


SENSOR_POINTS = {
    "t_east_c": "east_air_temperature_c",
    "t_center_c": "center_air_temperature_c",
    "t_west_c": "west_air_temperature_c",
    "rh_center_pct": "center_relative_humidity_pct",
    "canopy_temp_c": "center_canopy_temperature_c",
    "substrate_temp_c": "center_substrate_temperature_c",
    "substrate_moisture_pct": "center_substrate_moisture_pct",
    "co2_ppm": "center_co2_ppm",
    "out_temp_c": "outdoor_air_temperature_c",
    "out_rh_pct": "outdoor_relative_humidity_pct",
    "out_solar_w_m2": "outdoor_solar_irradiance_w_m2",
    "out_wind_m_s": "outdoor_wind_speed_m_s",
    "battery_soc_pct": "battery_soc_pct",
    "battery_power_kw": "battery_power_kw",
    "pv_power_kw": "pv_power_kw",
    "wind_power_kw": "wind_power_kw",
    "renewable_power_kw": "renewable_power_kw",
    "grid_power_kw": "grid_power_kw",
    "load_power_kw": "load_power_kw",
    "unserved_power_kw": "unserved_power_kw",
    "power_supply_ratio": "power_supply_ratio",
}

QUALITY_POINTS = {
    "t_east_quality": "east_air_temperature_c",
    "t_center_quality": "center_air_temperature_c",
    "t_west_quality": "west_air_temperature_c",
    "rh_center_quality": "center_relative_humidity_pct",
    "out_temp_quality": "outdoor_air_temperature_c",
    "out_rh_quality": "outdoor_relative_humidity_pct",
    "out_solar_quality": "outdoor_solar_irradiance_w_m2",
    "out_wind_quality": "outdoor_wind_speed_m_s",
}

SENSOR_DEFAULTS = {
    "t_east_c": 15.0,
    "t_center_c": 15.0,
    "t_west_c": 15.0,
    "rh_center_pct": 70.0,
    "canopy_temp_c": 15.0,
    "substrate_temp_c": 12.0,
    "substrate_moisture_pct": 55.0,
    "co2_ppm": 800.0,
    "out_temp_c": 0.0,
    "out_rh_pct": 60.0,
    "out_solar_w_m2": 0.0,
    "out_wind_m_s": 0.0,
    "battery_soc_pct": 70.0,
    "battery_power_kw": 0.0,
    "pv_power_kw": 0.0,
    "wind_power_kw": 0.0,
    "renewable_power_kw": 0.0,
    "grid_power_kw": 0.0,
    "load_power_kw": 0.0,
    "unserved_power_kw": 0.0,
    "power_supply_ratio": 1.0,
}


class GreenhouseSACEnv(gym.Env):
    """单棚温室连续控制环境，兼容 Stable-Baselines3 SAC。"""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    ACTION_NAMES = ("heater", "vent", "curtain", "fan")
    OBSERVATION_NAMES = (
        "t_east_norm",
        "t_center_norm",
        "t_west_norm",
        "rh_center_norm",
        "canopy_temp_norm",
        "substrate_temp_norm",
        "substrate_moisture_norm",
        "co2_norm",
        "out_temp_norm",
        "out_rh_norm",
        "solar_norm",
        "wind_norm",
        "heater_actual_norm",
        "vent_actual_norm",
        "curtain_actual_norm",
        "fan_actual_norm",
        "pred_10min_norm",
        "pred_30min_norm",
        "target_low_norm",
        "target_high_norm",
        "hour_sin",
        "hour_cos",
        "is_day",
        "transformer_quality",
    )

    def __init__(
        self,
        scenario_path: str | Path = BASE_SCENARIO,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        meta_path: str | Path = DEFAULT_META_PATH,
        episode_hours: float = 72.0,
        control_interval_minutes: int = 5,
        warmup_minutes: int = 60,
        use_transformer: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if control_interval_minutes < 1:
            raise ValueError("control_interval_minutes 必须大于等于 1")
        if episode_hours <= 0:
            raise ValueError("episode_hours 必须大于 0")

        self.scenario_path = Path(scenario_path)
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.episode_hours = float(episode_hours)
        self.control_interval_minutes = int(control_interval_minutes)
        self.warmup_minutes = int(warmup_minutes)
        self.use_transformer = bool(use_transformer)
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.ACTION_NAMES),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.OBSERVATION_NAMES),),
            dtype=np.float32,
        )

        self._vg: Optional[VirtualGreenhouse] = None
        self._cfg: Any = None
        self._predictor: Optional[TransformerPredictor] = None
        self._rng = random.Random()
        self._episode_seed = 0
        self._decision_step = 0
        self._max_decision_steps = max(
            1, int(round(self.episode_hours * 60 / self.control_interval_minutes))
        )
        self._last_valid: Dict[str, float] = {}
        self._actuals = {f"{name}_01": 0.0 for name in self.ACTION_NAMES}
        self._previous_targets = {name: 0.0 for name in self.ACTION_NAMES}
        self._latest_batch: Any = None
        self._latest_record: Dict[str, Any] = {}
        self._latest_prediction: Dict[str, Any] = {"status": "DISABLED"}
        self._last_info: Dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.close()

        options = options or {}
        self._episode_seed = int(
            options.get(
                "episode_seed",
                seed if seed is not None else self.np_random.integers(0, 2**31 - 1),
            )
        )
        self._rng = random.Random(self._episode_seed)
        self._decision_step = 0
        self._last_valid = {}
        self._actuals = {f"{name}_01": 0.0 for name in self.ACTION_NAMES}
        self._previous_targets = {name: 0.0 for name in self.ACTION_NAMES}

        self._cfg = load_config(self.scenario_path)
        total_hours = self.episode_hours + self.warmup_minutes / 60.0 + 0.25
        self._cfg.simulation.duration_hours = total_hours
        self._cfg.scenario_id = f"sac_single_shed_seed_{self._episode_seed}"

        # 每次 reset 都随机抽取天气组合，避免策略只记住一条固定曲线。
        weather_sets = (
            ("clear_cold",),
            ("overcast_humid",),
            ("clear_cold", "cold_snap"),
            ("overcast_series",),
        )
        scenes = tuple(options.get("weather_scenes", self._rng.choice(weather_sets)))
        base_temp = float(options.get("base_outdoor_temp_c", self._rng.uniform(0.0, 15.0)))
        self._cfg.weather.base_outdoor_temp_c = base_temp
        self._cfg.weather_schedule = sequential_schedule(scenes, total_hours)
        self._cfg.fault_schedule = list(options.get("fault_schedule", []))

        self._vg = VirtualGreenhouse(self._cfg, seed=self._episode_seed, evaluate=True)
        self._latest_batch = self._vg.reset()

        if self.use_transformer:
            if not self.model_path.exists() or not self.meta_path.exists():
                raise FileNotFoundError(
                    "未找到 Transformer 模型或元数据："
                    f"{self.model_path} / {self.meta_path}"
                )
            self._predictor = TransformerPredictor(self.model_path, self.meta_path)
            self._predictor.reset()
        else:
            self._predictor = None

        # 先用平滑规则控制器运行 60 分钟，填满 Transformer 输入窗口。
        for _ in range(self.warmup_minutes):
            values, _ = self._read_values(self._latest_batch)
            sim_hour = self._vg.sim_seconds / 3600.0
            targets = decide_targets("smooth", values, sim_hour, self._rng)
            result = self._vg.step(commands_from_targets(targets))
            self._actuals = feedback_actuals(result.feedback, self._actuals)
            self._previous_targets = {key: float(targets[key]) for key in self.ACTION_NAMES}
            self._latest_batch = result.observation
            self._latest_record, self._latest_prediction = self._update_record_and_prediction(
                self._latest_batch, self._previous_targets
            )
            if result.done:
                break

        observation = self._make_observation()
        info = self._make_info(
            requested_targets=self._previous_targets,
            safe_targets=self._previous_targets,
            safety_overrides=(),
            reward_components={},
        )
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._vg is None:
            raise RuntimeError("请先调用 reset()")

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (len(self.ACTION_NAMES),):
            raise ValueError(f"动作形状必须为 {(len(self.ACTION_NAMES),)}")
        if not np.all(np.isfinite(action_array)):
            action_array = np.nan_to_num(action_array, nan=0.0, posinf=1.0, neginf=-1.0)

        requested_targets = self._decode_action(action_array)
        safe_targets, safety_overrides = self._safety_filter(requested_targets)
        safe_movement = float(
            np.mean(
                [
                    abs(safe_targets[name] - self._previous_targets[name]) / 100.0
                    for name in self.ACTION_NAMES
                ]
            )
        )
        # 原始动作相对上一安全动作的偏移，避免策略依赖限速层掩盖大幅跳变。
        raw_request_movement = float(
            np.mean(
                [
                    abs(requested_targets[name] - self._previous_targets[name]) / 100.0
                    for name in self.ACTION_NAMES
                ]
            )
        )
        # 10%以下视为设备微小请求；超过部分按重叠程度连续惩罚。
        heater_overlap = max(0.0, requested_targets["heater"] - 10.0)
        vent_overlap = max(0.0, requested_targets["vent"] - 10.0)
        conflict_intensity = min(heater_overlap, vent_overlap) / 90.0
        unsafe_request = float(conflict_intensity > 0.0)

        energy_samples = []
        simulator_done = False
        for _ in range(self.control_interval_minutes):
            result = self._vg.step(commands_from_targets(safe_targets))
            self._actuals = feedback_actuals(result.feedback, self._actuals)
            self._latest_batch = result.observation
            self._latest_record, self._latest_prediction = self._update_record_and_prediction(
                self._latest_batch, safe_targets
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
        self._previous_targets = dict(safe_targets)
        energy_norm = float(np.mean(energy_samples)) if energy_samples else 0.0
        reward, reward_components = self._calculate_reward(
            safe_movement=safe_movement,
            raw_request_movement=raw_request_movement,
            energy_norm=energy_norm,
            unsafe_request=unsafe_request,
            conflict_intensity=conflict_intensity,
        )

        observation = self._make_observation()
        invalid_state = not np.all(np.isfinite(observation))
        temp = float(self._latest_record.get("t_center_c", 0.0))
        terminated = bool(invalid_state or temp < -20.0 or temp > 60.0)
        truncated = bool(
            simulator_done or self._decision_step >= self._max_decision_steps
        )
        info = self._make_info(
            requested_targets=requested_targets,
            safe_targets=safe_targets,
            safety_overrides=safety_overrides,
            reward_components=reward_components,
        )
        self._last_info = info
        return observation, reward, terminated, truncated, info

    def _read_values(self, batch: Any) -> Tuple[Dict[str, float], Dict[str, str]]:
        obs = observation_map(batch)
        enforce_sensor_bounds(obs)
        values: Dict[str, float] = {}
        for name, point_id in SENSOR_POINTS.items():
            value = obs_value(obs, point_id, self._last_valid)
            values[name] = float(SENSOR_DEFAULTS[name] if value is None else value)
        qualities = {
            name: str(quality_of(obs, point_id))
            for name, point_id in QUALITY_POINTS.items()
        }
        return values, qualities

    def _actual(self, action_name: str) -> float:
        return float(self._actuals.get(f"{action_name}_01", 0.0))

    def _update_record_and_prediction(
        self, batch: Any, targets: Mapping[str, float]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        values, qualities = self._read_values(batch)
        sim_hour = self._vg.sim_seconds / 3600.0 if self._vg is not None else 0.0
        start = datetime.fromisoformat(self._cfg.simulation.sim_start)
        timestamp = start + timedelta(seconds=self._vg.sim_seconds if self._vg else 0.0)
        threshold = float(self._cfg.targets.day_radiation_threshold_w_m2)

        record: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "sim_hour": sim_hour,
            "hour": timestamp.hour + timestamp.minute / 60.0,
            "is_day": int(values["out_solar_w_m2"] >= threshold),
            **values,
            **qualities,
            "heater_cmd_pct": float(targets["heater"]),
            "vent_cmd_pct": float(targets["vent"]),
            "curtain_cmd_pct": float(targets["curtain"]),
            "fan_cmd_pct": float(targets["fan"]),
            "heater_actual_pct": self._actual("heater"),
            "vent_actual_pct": self._actual("vent"),
            "curtain_actual_pct": self._actual("curtain"),
            "fan_actual_pct": self._actual("fan"),
        }
        if self._predictor is None:
            prediction = {
                "status": "DISABLED",
                "predictions_c": {
                    "10min": values["t_center_c"],
                    "30min": values["t_center_c"],
                },
                "input_quality": "DISABLED",
            }
        else:
            prediction = self._predictor.update(record)
        return record, prediction

    def _decode_action(self, action: np.ndarray) -> Dict[str, float]:
        clipped = np.clip(action, -1.0, 1.0)
        return {
            name: float((clipped[index] + 1.0) * 50.0)
            for index, name in enumerate(self.ACTION_NAMES)
        }

    def _target_band(self) -> Tuple[float, float]:
        is_day = bool(self._latest_record.get("is_day", 0))
        return (20.0, 25.0) if is_day else (8.0, 10.0)

    def _safety_filter(
        self, requested: Mapping[str, float]
    ) -> Tuple[Dict[str, float], Tuple[str, ...]]:
        safe = {name: float(np.clip(requested[name], 0.0, 100.0)) for name in self.ACTION_NAMES}
        overrides = []

        # 执行器变化率限制：避免策略在相邻控制周期剧烈抖动。
        max_delta = {"heater": 25.0, "vent": 20.0, "curtain": 15.0, "fan": 25.0}
        for name in self.ACTION_NAMES:
            low = self._previous_targets[name] - max_delta[name]
            high = self._previous_targets[name] + max_delta[name]
            limited = float(np.clip(safe[name], low, high))
            if abs(limited - safe[name]) > 1e-6:
                overrides.append(f"{name}_rate_limit")
            safe[name] = limited

        temp = float(self._latest_record.get("t_center_c", 15.0))
        rh = float(self._latest_record.get("rh_center_pct", 70.0))
        target_low, target_high = self._target_band()

        # 极端状态优先级最高。
        if temp < 3.0:
            safe.update({"heater": 100.0, "vent": 0.0, "curtain": 100.0, "fan": 0.0})
            overrides.append("extreme_cold_protection")
        elif temp > 32.0:
            safe["heater"] = 0.0
            safe["vent"] = max(safe["vent"], 80.0)
            safe["fan"] = max(safe["fan"], 50.0)
            safe["curtain"] = max(safe["curtain"], 80.0)
            overrides.append("extreme_heat_protection")

        # 正常控温时禁止高强度加热与通风对冲；高湿排湿允许小功率加热。
        if safe["heater"] > 10.0 and safe["vent"] > 10.0:
            if rh >= 90.0:
                safe["heater"] = min(safe["heater"], 30.0)
                overrides.append("dehumidify_override")
            elif temp <= (target_low + target_high) / 2.0:
                safe["vent"] = 0.0
                overrides.append("heating_priority")
            else:
                safe["heater"] = 0.0
                overrides.append("ventilation_priority")

        return safe, tuple(dict.fromkeys(overrides))

    @staticmethod
    def _distance_to_band(value: float, low: float, high: float) -> float:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    def _calculate_reward(
        self,
        safe_movement: float,
        raw_request_movement: float,
        energy_norm: float,
        unsafe_request: float,
        conflict_intensity: float,
    ) -> Tuple[float, Dict[str, float]]:
        temp = float(self._latest_record.get("t_center_c", 0.0))
        rh = float(self._latest_record.get("rh_center_pct", 70.0))
        target_low, target_high = self._target_band()
        temp_error = self._distance_to_band(temp, target_low, target_high)
        humidity_error = self._distance_to_band(rh, 60.0, 85.0)
        severe = float(temp < 3.0 or temp > 35.0)
        comfort_bonus = 0.20 if temp_error == 0.0 else 0.0

        components = {
            "comfort_bonus": comfort_bonus,
            "temperature_penalty": -4.0 * (temp_error / 5.0) ** 2,
            "humidity_penalty": -1.0 * (humidity_error / 20.0) ** 2,
            "energy_penalty": -0.40 * energy_norm,
            "movement_penalty": -0.15 * safe_movement,
            "raw_request_movement_penalty": -0.15 * raw_request_movement,
            "unsafe_request_penalty": -3.00 * conflict_intensity,
            "severe_state_penalty": -5.0 * severe,
        }
        reward = float(np.clip(sum(components.values()), -20.0, 1.0))
        components["temperature_error_c"] = temp_error
        components["humidity_error_pct"] = humidity_error
        components["energy_norm"] = energy_norm
        components["movement_norm"] = safe_movement
        components["raw_request_movement_norm"] = raw_request_movement
        components["conflict_intensity"] = conflict_intensity
        components["unsafe_request"] = unsafe_request
        components["reward_version"] = 2.0
        return reward, components

    def _prediction_value(self, horizon: str, fallback: float) -> float:
        predictions = self._latest_prediction.get("predictions_c", {})
        try:
            value = float(predictions.get(horizon, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if math.isfinite(value) else fallback

    @staticmethod
    def _temp_norm(value: float) -> float:
        return (value - 15.0) / 20.0

    def _make_observation(self) -> np.ndarray:
        r = self._latest_record
        center = float(r.get("t_center_c", 15.0))
        pred10 = self._prediction_value("10min", center)
        pred30 = self._prediction_value("30min", center)
        low, high = self._target_band()
        hour = float(r.get("hour", 0.0))
        quality = str(self._latest_prediction.get("input_quality", "UNKNOWN"))

        vector = np.asarray(
            [
                self._temp_norm(float(r.get("t_east_c", center))),
                self._temp_norm(center),
                self._temp_norm(float(r.get("t_west_c", center))),
                float(r.get("rh_center_pct", 70.0)) / 100.0,
                self._temp_norm(float(r.get("canopy_temp_c", center))),
                self._temp_norm(float(r.get("substrate_temp_c", center))),
                float(r.get("substrate_moisture_pct", 50.0)) / 100.0,
                (float(r.get("co2_ppm", 800.0)) - 800.0) / 800.0,
                float(r.get("out_temp_c", 0.0)) / 25.0,
                float(r.get("out_rh_pct", 60.0)) / 100.0,
                float(r.get("out_solar_w_m2", 0.0)) / 1000.0,
                float(r.get("out_wind_m_s", 0.0)) / 10.0,
                float(r.get("heater_actual_pct", 0.0)) / 100.0,
                float(r.get("vent_actual_pct", 0.0)) / 100.0,
                float(r.get("curtain_actual_pct", 0.0)) / 100.0,
                float(r.get("fan_actual_pct", 0.0)) / 100.0,
                self._temp_norm(pred10),
                self._temp_norm(pred30),
                self._temp_norm(low),
                self._temp_norm(high),
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                float(r.get("is_day", 0)),
                1.0 if quality == "GOOD" else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(np.nan_to_num(vector), -10.0, 10.0).astype(np.float32)

    def _make_info(
        self,
        requested_targets: Mapping[str, float],
        safe_targets: Mapping[str, float],
        safety_overrides: Iterable[str],
        reward_components: Mapping[str, float],
    ) -> Dict[str, Any]:
        low, high = self._target_band()
        center = float(self._latest_record.get("t_center_c", 0.0))
        energy_state = (
            self._vg.energy.state.to_dict()
            if self._vg is not None and hasattr(self._vg, "energy")
            else {}
        )
        return {
            "episode_seed": self._episode_seed,
            "decision_step": self._decision_step,
            "sim_hour": float(self._latest_record.get("sim_hour", 0.0)),
            "temperature_c": center,
            "relative_humidity_pct": float(
                self._latest_record.get("rh_center_pct", 0.0)
            ),
            "target_band_c": (low, high),
            "pred_10min_c": self._prediction_value("10min", center),
            "pred_30min_c": self._prediction_value("30min", center),
            "transformer_status": self._latest_prediction.get("status", "UNKNOWN"),
            "transformer_quality": self._latest_prediction.get(
                "input_quality", "UNKNOWN"
            ),
            "requested_targets_pct": dict(requested_targets),
            "safe_targets_pct": dict(safe_targets),
            "actuals_pct": {name: self._actual(name) for name in self.ACTION_NAMES},
            "safety_overrides": tuple(safety_overrides),
            "reward_components": dict(reward_components),
            "energy": energy_state,
        }

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        info = self._last_info or self._make_info(
            self._previous_targets, self._previous_targets, (), {}
        )
        return (
            f"step={info['decision_step']} hour={info['sim_hour']:.2f} "
            f"T={info['temperature_c']:.2f}C RH={info['relative_humidity_pct']:.1f}% "
            f"safe={info['safe_targets_pct']}"
        )

    def close(self) -> None:
        if self._vg is not None:
            try:
                self._vg.stop()
            except Exception:
                pass
            self._vg = None


if __name__ == "__main__":
    env = GreenhouseSACEnv(episode_hours=2.0, render_mode="ansi")
    try:
        obs, info = env.reset(seed=42)
        print("observation_shape=", obs.shape)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            print(env.render(), "reward=", round(reward, 3))
            if terminated or truncated:
                break
    finally:
        env.close()
