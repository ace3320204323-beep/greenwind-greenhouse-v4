"""绿风润耕：单棚 Transformer 训练数据生成器 v0.2。

相较 v0.1 的关键修复：
1. 只通过 VirtualGreenhouse.reset/step 公共接口交互；
2. 不再额外调用 weather.step，避免每个周期重复推进天气随机状态；
3. 天气场景按时间顺序分段，不再重叠；
4. 模型输入来自传感器 Observation，而不是仿真内部真值；
5. 明确字段名称和单位，区分相对湿度与绝对湿度；
6. 保存控制目标、设备实际反馈、数据质量和仿真真值审计字段；
7. 使用多种安全控制策略增加动作覆盖范围。

默认正式运行：
    python tools/ai_pipeline/generate_data_v2.py

快速冒烟测试：
    python tools/ai_pipeline/generate_data_v2.py --days 1 --groups 2 \
        --out tools/ai_pipeline/data/dataset_v2_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from virtual_greenhouse.config import load_config
from virtual_greenhouse.runner import VirtualGreenhouse
from virtual_greenhouse.schemas import (
    ActionType,
    CommandSource,
    ControlCommand,
    ControlMode,
    Quality,
    new_id,
)


BASE_SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "virtual_greenhouse"
    / "scenarios"
    / "shandong_winter_strawberry.json"
)

USABLE_QUALITIES = {
    Quality.GOOD.value,
    Quality.SUSPECT.value,
    # 仅当 enforce_sensor_bounds 已将数值裁剪到物理范围后使用。
    Quality.OUT_OF_RANGE.value,
}


@dataclass(frozen=True)
class GroupSpec:
    base_temp_c: float
    scenes: tuple[str, ...]
    policy: str
    heater_efficiency_drop_hour: float | None = None


GROUP_SPECS: tuple[GroupSpec, ...] = (
    GroupSpec(0.0, ("clear_cold",), "rule"),
    GroupSpec(0.0, ("clear_cold", "cold_snap"), "smooth", 84.0),
    GroupSpec(5.0, ("clear_cold",), "smooth"),
    GroupSpec(5.0, ("overcast_humid",), "probe"),
    GroupSpec(9.0, ("clear_cold",), "rule"),
    GroupSpec(9.0, ("clear_cold", "overcast_humid"), "probe"),
    GroupSpec(13.0, ("clear_cold",), "smooth"),
    GroupSpec(13.0, ("clear_cold", "cold_snap"), "rule", 84.0),
    GroupSpec(3.0, ("overcast_series",), "probe"),
    GroupSpec(7.0, ("clear_cold", "cold_snap", "clear_cold"), "smooth"),
    GroupSpec(11.0, ("overcast_humid", "clear_cold"), "rule"),
    GroupSpec(15.0, ("clear_cold",), "probe"),
)


FIELDNAMES = [
    # 数据来源与时间
    "data_source", "scenario_id", "greenhouse_id", "group_id", "seed",
    "sim_step", "sim_hour", "timestamp", "hour", "is_day", "weather_scene",
    # 传感器观测：模型实际可读
    "t_east_c", "t_center_c", "t_west_c", "rh_center_pct",
    "canopy_temp_c", "substrate_temp_c", "substrate_moisture_pct", "co2_ppm",
    "out_temp_c", "out_rh_pct", "out_solar_w_m2", "out_wind_m_s",
    # 质量标记
    "t_east_quality", "t_center_quality", "t_west_quality", "rh_center_quality",
    "out_temp_quality", "out_rh_quality", "out_solar_quality", "out_wind_quality",
    # 本周期控制目标与执行后的实际反馈
    "heater_cmd_pct", "vent_cmd_pct", "curtain_cmd_pct", "fan_cmd_pct",
    "heater_actual_pct", "vent_actual_pct", "curtain_actual_pct", "fan_actual_pct",
    "control_policy",
    # 仅用于事后审计，禁止作为部署时模型输入
    "truth_t_center_c", "truth_abs_hum_center_g_m3", "heater_efficiency_truth",
]


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def sequential_schedule(scenes: Iterable[str], duration_hours: float) -> list[dict]:
    """把天气场景顺序、无重叠地覆盖整个仿真周期。"""
    names = tuple(scenes) or ("clear_cold",)
    segment = duration_hours / len(names)
    schedule: list[dict] = []
    for index, scene in enumerate(names):
        start = index * segment
        end = duration_hours if index == len(names) - 1 else (index + 1) * segment
        schedule.append({"start_hour": start, "end_hour": end, "scene": scene})
    return schedule


def active_scene(schedule: list[dict], sim_hour: float) -> str:
    for segment in schedule:
        if float(segment["start_hour"]) <= sim_hour < float(segment["end_hour"]):
            return str(segment["scene"])
    return str(schedule[-1]["scene"]) if schedule else "clear_cold"


def observation_map(batch) -> dict[str, tuple[float | None, str]]:
    """point_id -> (value, quality)。保留缺失值，不偷看仿真真值补齐。"""
    result: dict[str, tuple[float | None, str]] = {}
    for obs in batch.observations:
        quality = obs.quality.value if hasattr(obs.quality, "value") else str(obs.quality)
        result[obs.point_id] = (obs.value, quality)
    return result


def enforce_sensor_bounds(observations: dict[str, tuple[float | None, str]]) -> None:
    """在AI接入层执行物理边界检查，不修改已封版的传感器模型。

    传感器真值可能处于物理范围内，但叠加噪声、偏置和量化后仍可能轻微越界。
    对部署侧输入采用统一规则：裁剪到合法范围，并把质量标记改为
    OUT_OF_RANGE，供后续模型掩码和审计使用。
    """
    bounds = {
        "center_relative_humidity_pct": (0.0, 100.0),
        "outdoor_relative_humidity_pct": (0.0, 100.0),
        "outdoor_solar_irradiance_w_m2": (0.0, None),
        "outdoor_wind_speed_m_s": (0.0, None),
    }
    for point_id, (low, high) in bounds.items():
        value, quality = observations.get(point_id, (None, Quality.MISSING.value))
        if value is None:
            continue
        corrected = float(value)
        out_of_range = corrected < low or (high is not None and corrected > high)
        if not out_of_range:
            continue
        corrected = max(low, corrected)
        if high is not None:
            corrected = min(high, corrected)
        observations[point_id] = (corrected, Quality.OUT_OF_RANGE.value)


def obs_value(
    observations: dict[str, tuple[float | None, str]],
    point_id: str,
    last_valid: dict[str, float],
) -> float | None:
    value, quality = observations.get(point_id, (None, Quality.MISSING.value))
    if value is not None and quality in USABLE_QUALITIES:
        last_valid[point_id] = float(value)
        return float(value)
    # 控制器使用最近有效值；CSV 中仍会单独保存原始质量标记。
    return last_valid.get(point_id)


def quality_of(observations: dict[str, tuple[float | None, str]], point_id: str) -> str:
    return observations.get(point_id, (None, Quality.MISSING.value))[1]


def make_command(device_id: str, value: float, action_type: ActionType) -> ControlCommand:
    return ControlCommand(
        command_id=new_id("cmd"),
        trace_id=new_id("trace"),
        greenhouse_id="GH001",
        device_id=device_id,
        action_type=action_type,
        target_value=clip(float(value)),
        unit="%",
        issued_at="",
        effective_at="",
        expires_at="",
        control_mode=ControlMode.RULE,
        source=CommandSource.LOCAL_CONTROLLER,
        model_version="data-generator-v0.2",
        safety_checked=True,
    )


def decide_targets(
    policy: str,
    values: dict[str, float | None],
    sim_hour: float,
    rng: random.Random,
) -> dict[str, float]:
    """生成有变化但受安全约束的控制目标，用于覆盖不同动作状态。"""
    temp = values.get("t_center_c")
    rh = values.get("rh_center_pct")
    solar = values.get("out_solar_w_m2")
    temp = 15.0 if temp is None else float(temp)
    rh = 70.0 if rh is None else float(rh)
    solar = 0.0 if solar is None else float(solar)

    is_day = solar >= 50.0
    target_mid = 22.5 if is_day else 9.0

    if policy == "smooth":
        heater = clip((target_mid - temp) * 14.0)
        vent = clip((temp - 24.0) * 18.0)
    elif policy == "probe":
        # 只在安全区间内叠加低幅、缓慢探测信号，提高动作-响应可辨识性。
        base_heater = clip((target_mid - temp) * 12.0)
        probe = 10.0 * math.sin(2.0 * math.pi * sim_hour / 3.0) + rng.uniform(-2.0, 2.0)
        heater = clip(base_heater + probe) if 7.0 <= temp <= 24.0 else base_heater
        vent = clip((temp - 23.0) * 15.0)
    else:
        heater = 100.0 if temp < target_mid - 2.0 else (50.0 if temp < target_mid else 0.0)
        vent = 40.0 if temp > 25.0 else 0.0

    curtain = 0.0 if is_day else 100.0
    fan = 50.0 if (temp > 26.0 or rh > 90.0) else 0.0

    # 安全互斥：仅高湿排湿时允许低功率加热与通风协同。
    dehumidifying = rh > 90.0
    if vent > 0.0 and heater > 0.0:
        if dehumidifying:
            heater = min(heater, 30.0)
        elif temp >= target_mid:
            heater = 0.0
        else:
            vent = 0.0

    return {
        "heater": round(clip(heater), 3),
        "vent": round(clip(vent), 3),
        "curtain": round(clip(curtain), 3),
        "fan": round(clip(fan), 3),
    }


def commands_from_targets(targets: dict[str, float]) -> list[ControlCommand]:
    return [
        make_command("heater_01", targets["heater"], ActionType.SET_POWER),
        make_command("vent_01", targets["vent"], ActionType.SET_POSITION),
        make_command("curtain_01", targets["curtain"], ActionType.SET_POSITION),
        make_command("fan_01", targets["fan"], ActionType.SET_POWER),
    ]


def feedback_actuals(feedbacks, previous: dict[str, float]) -> dict[str, float]:
    values = dict(previous)
    for feedback in feedbacks:
        values[feedback.device_id] = float(feedback.actual_value)
    return values


def run_group(spec: GroupSpec, group_id: int, days: int, seed: int) -> list[dict]:
    duration_hours = float(days * 24)
    cfg = load_config(BASE_SCENARIO)
    cfg.scenario_id = f"{cfg.scenario_id}_ai_g{group_id:02d}"
    cfg.weather.base_outdoor_temp_c = spec.base_temp_c
    cfg.simulation.duration_hours = duration_hours
    cfg.weather_schedule = sequential_schedule(spec.scenes, duration_hours)
    cfg.fault_schedule = []
    if spec.heater_efficiency_drop_hour is not None:
        fault_hour = min(float(spec.heater_efficiency_drop_hour), duration_hours * 0.75)
        cfg.fault_schedule.append({
            "start_hour": fault_hour,
            "event_type": "heater_degraded",
            "detail": {"device_id": "heater_01", "efficiency": 0.75},
        })

    vg = VirtualGreenhouse(cfg, seed=seed, evaluate=True)
    policy_rng = random.Random(seed ^ 0xA17E)
    rows: list[dict] = []
    last_valid: dict[str, float] = {}
    actuals = {
        "heater_01": 0.0,
        "vent_01": 0.0,
        "curtain_01": 0.0,
        "fan_01": 0.0,
    }

    try:
        current_batch = vg.reset()
        steps = int(duration_hours * 3600.0 / cfg.simulation.dt_seconds)

        for _ in range(steps):
            current_obs = observation_map(current_batch)
            enforce_sensor_bounds(current_obs)
            current_values = {
                "t_center_c": obs_value(current_obs, "center_air_temperature_c", last_valid),
                "rh_center_pct": obs_value(current_obs, "center_relative_humidity_pct", last_valid),
                "out_solar_w_m2": obs_value(current_obs, "outdoor_solar_irradiance_w_m2", last_valid),
            }
            targets = decide_targets(spec.policy, current_values, vg.sim_seconds / 3600.0, policy_rng)
            result = vg.step(commands_from_targets(targets))
            actuals = feedback_actuals(result.feedback, actuals)

            batch = result.observation
            observations = observation_map(batch)
            enforce_sensor_bounds(observations)
            values = {
                "t_east_c": obs_value(observations, "east_air_temperature_c", last_valid),
                "t_center_c": obs_value(observations, "center_air_temperature_c", last_valid),
                "t_west_c": obs_value(observations, "west_air_temperature_c", last_valid),
                "rh_center_pct": obs_value(observations, "center_relative_humidity_pct", last_valid),
                "canopy_temp_c": obs_value(observations, "center_canopy_temperature_c", last_valid),
                "substrate_temp_c": obs_value(observations, "center_substrate_temperature_c", last_valid),
                "substrate_moisture_pct": obs_value(observations, "center_substrate_moisture_pct", last_valid),
                "co2_ppm": obs_value(observations, "center_co2_ppm", last_valid),
                "out_temp_c": obs_value(observations, "outdoor_air_temperature_c", last_valid),
                "out_rh_pct": obs_value(observations, "outdoor_relative_humidity_pct", last_valid),
                "out_solar_w_m2": obs_value(observations, "outdoor_solar_irradiance_w_m2", last_valid),
                "out_wind_m_s": obs_value(observations, "outdoor_wind_speed_m_s", last_valid),
            }

            sim_hour = vg.sim_seconds / 3600.0
            timestamp = datetime.fromisoformat(cfg.simulation.sim_start) + timedelta(seconds=vg.sim_seconds)
            hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
            solar = values["out_solar_w_m2"] or 0.0
            truth = result.evaluation_truth

            row = {
                "data_source": "SIMULATION",
                "scenario_id": cfg.scenario_id,
                "greenhouse_id": cfg.greenhouse_id,
                "group_id": group_id,
                "seed": seed,
                "sim_step": result.observation.sim_step,
                "sim_hour": round(sim_hour, 6),
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "hour": round(hour, 6),
                "is_day": int(solar >= cfg.targets.day_radiation_threshold_w_m2),
                "weather_scene": active_scene(cfg.weather_schedule, sim_hour),
                **values,
                "t_east_quality": quality_of(observations, "east_air_temperature_c"),
                "t_center_quality": quality_of(observations, "center_air_temperature_c"),
                "t_west_quality": quality_of(observations, "west_air_temperature_c"),
                "rh_center_quality": quality_of(observations, "center_relative_humidity_pct"),
                "out_temp_quality": quality_of(observations, "outdoor_air_temperature_c"),
                "out_rh_quality": quality_of(observations, "outdoor_relative_humidity_pct"),
                "out_solar_quality": quality_of(observations, "outdoor_solar_irradiance_w_m2"),
                "out_wind_quality": quality_of(observations, "outdoor_wind_speed_m_s"),
                "heater_cmd_pct": targets["heater"],
                "vent_cmd_pct": targets["vent"],
                "curtain_cmd_pct": targets["curtain"],
                "fan_cmd_pct": targets["fan"],
                "heater_actual_pct": round(actuals.get("heater_01", 0.0), 3),
                "vent_actual_pct": round(actuals.get("vent_01", 0.0), 3),
                "curtain_actual_pct": round(actuals.get("curtain_01", 0.0), 3),
                "fan_actual_pct": round(actuals.get("fan_01", 0.0), 3),
                "control_policy": spec.policy,
                "truth_t_center_c": truth.zone_center_air_temperature_c,
                "truth_abs_hum_center_g_m3": truth.zone_center_absolute_humidity_g_m3,
                "heater_efficiency_truth": round(vg.plant.heater_efficiency, 4),
            }
            rows.append(row)
            current_batch = batch
            if result.done:
                break
    finally:
        vg.stop()

    return rows


def validate_rows(rows: list[dict], expected_groups: int) -> None:
    if not rows:
        raise RuntimeError("未生成任何数据")
    groups = {int(row["group_id"]) for row in rows}
    if len(groups) != expected_groups:
        raise RuntimeError(f"组数错误：期望 {expected_groups}，实际 {len(groups)}")
    for field in ("t_center_c", "rh_center_pct", "out_temp_c", "out_solar_w_m2"):
        non_missing = [row[field] for row in rows if row[field] is not None]
        if not non_missing:
            raise RuntimeError(f"关键字段全部缺失：{field}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成单棚 Transformer 训练数据 v0.2")
    parser.add_argument("--days", type=int, default=7, help="每组仿真天数")
    parser.add_argument("--groups", type=int, default=12, help="仿真组数，最大12")
    parser.add_argument("--seed0", type=int, default=100, help="第一组随机种子")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "dataset_v2.csv"),
        help="输出CSV路径",
    )
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("--days 必须大于0")
    group_count = min(max(1, args.groups), len(GROUP_SPECS))
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for group_id, spec in enumerate(GROUP_SPECS[:group_count]):
        seed = args.seed0 + group_id
        print(
            f"生成组 {group_id + 1}/{group_count}: base={spec.base_temp_c}℃ "
            f"scenes={list(spec.scenes)} policy={spec.policy} seed={seed}"
        )
        group_rows = run_group(spec, group_id, args.days, seed)
        all_rows.extend(group_rows)
        print(f"  完成：{len(group_rows)} 行")

    validate_rows(all_rows, group_count)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n数据已写入：{output_path}")
    print(f"总行数：{len(all_rows)}，组数：{group_count}")
    print("注意：truth_* 与 heater_efficiency_truth 仅供审计，不得作为模型输入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
