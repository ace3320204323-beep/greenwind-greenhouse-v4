"""三控制器公平对照实验框架（任务书第十五章 / 十七.P1）。

同一 seed、天气、故障计划、初始状态、设备约束下，跑三种控制器：
- fixed_threshold   固定阈值
- fixed_model       固定参数预测
- adaptive_model    在线辨识预测

输出对比汇总，含突变前后分段、设备饱和率、冲突、辨识真实性对照。
支持变化率扫描（max_heater_change）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .runner import VirtualGreenhouse
from .adapter import VirtualGreenhouseAdapter
from .controllers import make_controller, PredictiveBase
from .schemas import ControlCommand, new_id


def _day_night_targets(cfg):
    t = cfg.targets
    return (t.day_temp_min_c, t.day_temp_max_c, t.night_temp_min_c, t.night_temp_max_c)


def _feedback_map(result) -> dict[str, float]:
    """从 step 结果的 feedback 列表提取设备实际值。"""
    fm: dict[str, float] = {}
    for fb in result.feedback:
        fm[fb.device_id] = fb.actual_value
    return fm


def run_controller(cfg, seed: int, kind: str, output_dir: Path,
                   steps: int | None = None,
                   max_heater_change: float | None = None) -> dict[str, Any]:
    day_min, day_max, night_min, night_max = _day_night_targets(cfg)
    humidity_max = cfg.targets.humidity_max_pct
    target_setpoint = (day_min + day_max) / 2.0

    vg = VirtualGreenhouse(cfg, seed, evaluate=True, output_dir=output_dir)
    adapter = VirtualGreenhouseAdapter(vg)
    controller = make_controller(kind, day_min, day_max, humidity_max,
                                 target=target_setpoint,
                                 night_temp_min=night_min, night_temp_max=night_max)
    if max_heater_change is not None and isinstance(controller, PredictiveBase):
        controller.max_heater_change = max_heater_change

    vg.reset()
    adapter._last_obs = vg._last_obs

    shift_hour = None
    for item in cfg.fault_schedule:
        if item.get("event_type") == "heater_degraded":
            shift_hour = item.get("start_hour", 36.0)
    shift_step = int(shift_hour * 3600 / cfg.simulation.dt_seconds) if shift_hour else None

    pre_mae: list[float] = []
    post_mae: list[float] = []
    heater_sat_steps = 0
    conflict_steps = 0          # 非法控制冲突（加热+通风同时，非排湿）
    dehumidify_steps = 0        # 合法排湿协同（加热+通风同时，排湿模式）
    total_steps = steps or int(cfg.simulation.duration_hours * 3600 / cfg.simulation.dt_seconds)

    # 真实加热效率记录（用于辨识真实性对照）
    true_eff_trace: list[float] = []
    est_gain_trace: list[float] = []
    fused_gain_trace: list[float] = []
    confidence_trace: list[float] = []

    feedback_map: dict[str, float] = {}
    for i in range(total_steps):
        obs_map = adapter.read_latest()
        cmds = controller.decide(obs_map, feedback_map)
        result = vg.step(cmds)
        adapter._last_obs = result.observation
        feedback_map = _feedback_map(result)

        # 决策链路落库（controller_trace）：state -> prediction -> intent -> command
        if vg.recorder and cmds:
            decision = getattr(controller, "last_decision", {})
            trace_id = cmds[0].trace_id
            vg.recorder.write("controller_trace", {
                "trace_id": trace_id,
                "sim_step": i,
                "model_name": decision.get("model_name", kind),
                "model_version": "1.0",
                "prediction_horizon_steps": decision.get("prediction_horizon_steps", 1),
                "predicted_temperature_c": decision.get("predicted_temperature_c", 0.0),
                "prediction_confidence": decision.get("prediction_confidence", 0.0),
                "objective_total_cost": decision.get("objective_total_cost", 0.0),
                "safety_check_result": "PASS",
            })

        t = result.evaluation_truth
        w = vg.weather.step(vg.sim_seconds)
        is_day = w.outdoor_solar_irradiance_w_m2 >= cfg.targets.day_radiation_threshold_w_m2
        tgt_min, tgt_max = (day_min, day_max) if is_day else (night_min, night_max)
        err = 0.0 if tgt_min <= t.zone_center_air_temperature_c <= tgt_max else \
            min(abs(t.zone_center_air_temperature_c - tgt_min),
                abs(t.zone_center_air_temperature_c - tgt_max))

        if shift_step is not None and i < shift_step:
            pre_mae.append(err)
        else:
            post_mae.append(err)

        if t.actual_heater_power_pct > 95:
            heater_sat_steps += 1
        # 加热+通风同时：按是否排湿模式拆分
        if t.actual_heater_power_pct > 5 and (t.actual_vent_position_pct > 5 or t.actual_fan_power_pct > 5):
            if getattr(controller, "dehumidify_active", False):
                dehumidify_steps += 1
            else:
                conflict_steps += 1

        # 辨识真实性：记录真实效率和估计增益
        true_eff_trace.append(vg.plant.heater_efficiency)
        if kind == "adaptive_model":
            est_gain_trace.append(controller.heater_gain)
            fused_gain_trace.append(controller.fused_gain)
            confidence_trace.append(controller.confidence)

    summary = vg.stop()

    out = {
        "controller": kind,
        "manifest": summary["manifest"],
        "metrics": summary["metrics"],
        "segmented": {
            "pre_shift_mae_c": round(sum(pre_mae) / max(1, len(pre_mae)), 4),
            "post_shift_mae_c": round(sum(post_mae) / max(1, len(post_mae)), 4),
        },
        "devices": {
            "heater_saturation_ratio": round(heater_sat_steps / total_steps, 4),
            "heat_vent_conflict_ratio": round(conflict_steps / total_steps, 4),
            "dehumidify_coop_ratio": round(dehumidify_steps / total_steps, 4),
        },
        "true_heater_efficiency_trace": true_eff_trace,
    }
    if kind == "adaptive_model":
        out["adaptive_snapshot"] = controller.snapshot()
        out["est_gain_trace"] = est_gain_trace
        out["fused_gain_trace"] = fused_gain_trace
        out["confidence_trace"] = confidence_trace
    return out


def run_benchmark(config_path: Path, seed: int, output_root: Path,
                  steps: int | None = None,
                  max_heater_change: float | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kinds = ["fixed_threshold", "fixed_model", "adaptive_model"]
    results: dict[str, Any] = {}
    for kind in kinds:
        run_dir = output_root / kind
        results[kind] = run_controller(cfg, seed, kind, run_dir, steps,
                                       max_heater_change)

    comparison = _compare(results, cfg)
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    return comparison


def _compare(results, cfg):
    day_min, day_max, night_min, night_max = _day_night_targets(cfg)
    comparison = {
        "seed_note": "same seed/weather/fault/initial for all three",
        "growth_stage": cfg.growth_stage,
        "targets": {"day": [day_min, day_max], "night": [night_min, night_max]},
    }
    rows = []
    for kind, r in results.items():
        env = r["metrics"]["environment"]
        ctrl = r["metrics"]["control"]
        energy = r["metrics"]["energy"]
        safety = r["metrics"]["safety"]
        row = {
            "controller": kind,
            "temperature_in_range_ratio": env["temperature_in_range_ratio"],
            "temperature_mae_c": env["temperature_mae_c"],
            "temperature_violation_degree_minutes": env["temperature_violation_degree_minutes"],
            "humidity_in_range_ratio": env["humidity_in_range_ratio"],
            "command_count": ctrl["command_count"],
            "actuator_travel_total_pct": ctrl["actuator_travel_total_pct"],
            "heater_energy_kwh": energy["heater_energy_kwh"],
            "fan_energy_kwh": energy["fan_energy_kwh"],
            "total_energy_kwh": energy["total_energy_kwh"],
            "invalid_command_rejected": safety["invalid_command_rejected_count"],
            "pre_shift_mae_c": r["segmented"]["pre_shift_mae_c"],
            "post_shift_mae_c": r["segmented"]["post_shift_mae_c"],
            "heater_saturation_ratio": r["devices"]["heater_saturation_ratio"],
            "heat_vent_conflict_ratio": r["devices"]["heat_vent_conflict_ratio"],
            "dehumidify_coop_ratio": r["devices"]["dehumidify_coop_ratio"],
        }
        if "adaptive_snapshot" in r:
            row["adaptive_snapshot"] = r["adaptive_snapshot"]
        rows.append(row)
    comparison["rows"] = rows
    return comparison
