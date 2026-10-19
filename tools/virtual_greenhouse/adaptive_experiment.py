"""独立自适应能力实验（P1-adaptive-experiment）。

在温和可控场景（突变前后目标均物理可达、加热不长期饱和）下，验证在线辨识
（控制器 C）相对固定参数预测（控制器 B）是否真正产生控制收益。

对比对象：
- B：固定参数预测控制器（heater_gain 恒为 0.45）
- C：带温差项 RLS 在线辨识预测控制器
- Oracle：诊断控制器，直接读取仿真器真实加热效率（仅用于判断理论上限，不作为成果）

实验设计（用户最终方案）：
- 72 小时；36h 加热效率 1.00 → 0.75；关闭其它故障。
- 多随机种子 seed=42/43/44/45/46。
- 突变时刻为 C 设置 baseline_gain（突变前稳定估计值），之后每步计算归一化增益
  normalized_gain = 当前估计值 / baseline_gain（理论上应逐步接近 0.75）。
- 分段统计：突变前 / 突变后 0-1h / 1-3h / 3-6h / 6-12h / 全时段。
- 恢复时间统一定义：突变后，连续 30 分钟温度重新进入目标范围，或滚动 MAE
  恢复到突变前基线的 120% 以内所需时间。

输出：
- 每个 seed 一份 results 字典，含分段指标 + 归一化增益时间序列 + 恢复时间。
- 汇总 comparison 供验收判断（不调场景让 C 获胜）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .runner import VirtualGreenhouse
from .adapter import VirtualGreenhouseAdapter
from .controllers import make_controller, AdaptiveController, OracleController, PredictiveBase


# ---------------------------------------------------------------------------
# 指标辅助
# ---------------------------------------------------------------------------

def _in_range(truth, tgt_min, tgt_max) -> bool:
    return tgt_min <= truth.zone_center_air_temperature_c <= tgt_max


def _err(truth, tgt_min, tgt_max) -> float:
    t = truth.zone_center_air_temperature_c
    if tgt_min <= t <= tgt_max:
        return 0.0
    return min(abs(t - tgt_min), abs(t - tgt_max))


def _violation_degmin(truth, tgt_min, tgt_max, dt_seconds) -> float:
    """越界度·分：越界度数 × 持续时间（分钟）。"""
    e = _err(truth, tgt_min, tgt_max)
    return e * dt_seconds / 60.0


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------

def run_single(cfg, seed: int, kind: str) -> dict[str, Any]:
    """跑一个 seed 一个控制器，返回分段指标与辨识/增益轨迹。"""
    day_min, day_max = cfg.targets.day_temp_min_c, cfg.targets.day_temp_max_c
    night_min, night_max = cfg.targets.night_temp_min_c, cfg.targets.night_temp_max_c
    humidity_max = cfg.targets.humidity_max_pct
    target_setpoint = (day_min + day_max) / 2.0
    dt = cfg.simulation.dt_seconds

    # 突变时刻
    shift_hour = None
    for item in cfg.fault_schedule:
        if item.get("event_type") == "heater_degraded":
            shift_hour = item.get("start_hour", 36.0)
    shift_step = int(shift_hour * 3600 / dt) if shift_hour else None

    vg = VirtualGreenhouse(cfg, seed, evaluate=True)
    adapter = VirtualGreenhouseAdapter(vg)
    controller = make_controller(kind, day_min, day_max, humidity_max, target=target_setpoint,
                                 night_temp_min=night_min, night_temp_max=night_max)

    vg.reset()
    adapter._last_obs = vg._last_obs

    total_steps = int(cfg.simulation.duration_hours * 3600 / dt)

    # 分段累计器
    segments = {
        "pre": {"mae_sum": 0.0, "n": 0, "violation_degmin": 0.0, "energy_kwh": 0.0,
                "sat_steps": 0, "travel_pct": 0.0},
        "0_1h": {"mae_sum": 0.0, "n": 0, "violation_degmin": 0.0, "energy_kwh": 0.0,
                 "sat_steps": 0, "travel_pct": 0.0},
        "1_3h": {"mae_sum": 0.0, "n": 0, "violation_degmin": 0.0, "energy_kwh": 0.0,
                 "sat_steps": 0, "travel_pct": 0.0},
        "3_6h": {"mae_sum": 0.0, "n": 0, "violation_degmin": 0.0, "energy_kwh": 0.0,
                 "sat_steps": 0, "travel_pct": 0.0},
        "6_12h": {"mae_sum": 0.0, "n": 0, "violation_degmin": 0.0, "energy_kwh": 0.0,
                  "sat_steps": 0, "travel_pct": 0.0},
    }

    # 归一化增益轨迹 / 真实效率轨迹
    norm_gain_trace: list[float | None] = []
    est_gain_trace: list[float] = []
    true_eff_trace: list[float] = []

    # 恢复时间相关：滚动 MAE（30 分钟窗口=30 步）、突变前基线 MAE
    roll_mae_window = 30
    roll_mae_queue: list[float] = []
    pre_mae_vals: list[float] = []   # 突变前所有步的 err（用于算基线）
    recovery_step: int | None = None

    feedback_map: dict[str, float] = {}
    prev_heater_actual = 0.0

    # 突变前稳定估计增益（用于 baseline_gain）——取突变前最后 6h 的估计均值
    pre_gain_queue: list[float] = []

    for i in range(total_steps):
        obs_map = adapter.read_latest()
        cmds = controller.decide(obs_map, feedback_map)
        result = vg.step(cmds)
        adapter._last_obs = result.observation
        feedback_map = {fb.device_id: fb.actual_value for fb in result.feedback}

        truth = result.evaluation_truth
        w = vg.weather.step(vg.sim_seconds)
        is_day = w.outdoor_solar_irradiance_w_m2 >= cfg.targets.day_radiation_threshold_w_m2
        tgt_min, tgt_max = (day_min, day_max) if is_day else (night_min, night_max)

        err = _err(truth, tgt_min, tgt_max)
        viol = _violation_degmin(truth, tgt_min, tgt_max, dt)
        heater_energy = truth.actual_heater_power_pct / 100.0 * 30.0 * dt / 3600.0

        # 执行器行程（实际值差）
        if i > 0:
            travel = (abs(truth.actual_heater_power_pct - prev_heater_actual)
                      + abs(truth.actual_vent_position_pct - 0.0))
        else:
            travel = 0.0
        prev_heater_actual = truth.actual_heater_power_pct

        # 分段归属
        if shift_step is not None and i < shift_step:
            seg = "pre"
            pre_mae_vals.append(err)
        elif shift_step is not None:
            d = i - shift_step
            if d < 60:
                seg = "0_1h"
            elif d < 180:
                seg = "1_3h"
            elif d < 360:
                seg = "3_6h"
            elif d < 720:
                seg = "6_12h"
            else:
                seg = "6_12h"  # 12h 之后并入最后段（或可扩展）
        else:
            seg = "pre"

        s = segments[seg]
        s["mae_sum"] += err
        s["n"] += 1
        s["violation_degmin"] += viol
        s["energy_kwh"] += heater_energy
        if truth.actual_heater_power_pct > 95:
            s["sat_steps"] += 1
        s["travel_pct"] += travel

        # 恢复时间：滚动 MAE 恢复到突变前基线的 120% 以内
        if shift_step is not None and i >= shift_step and recovery_step is None:
            roll_mae_queue.append(err)
            if len(roll_mae_queue) > roll_mae_window:
                roll_mae_queue.pop(0)
            if len(pre_mae_vals) > 0:
                baseline_mae = sum(pre_mae_vals) / len(pre_mae_vals)
                roll = sum(roll_mae_queue) / len(roll_mae_queue)
                # 允许基线为 0（前段完全达标）时，用当前滚动 MAE 与突变后首段对比
                threshold = baseline_mae * 1.2 if baseline_mae > 1e-6 else max(roll_mae_window and 0.1, 0.1)
                if roll <= threshold:
                    recovery_step = i - shift_step

        # 记录轨迹
        true_eff_trace.append(vg.plant.heater_efficiency)
        if isinstance(controller, AdaptiveController):
            est_gain_trace.append(controller.heater_gain)
            norm_gain_trace.append(controller.normalized_gain)
            # 突变前滚动记录 gain（用于 baseline）
            if shift_step is not None and i < shift_step:
                pre_gain_queue.append(controller.heater_gain)

        # 突变时刻：为 C 设置 baseline_gain（突变前最后 6h 的稳定估计均值）
        if isinstance(controller, AdaptiveController) and shift_step is not None and i == shift_step - 1:
            if len(pre_gain_queue) >= 60:
                baseline = sum(pre_gain_queue[-360:]) / len(pre_gain_queue[-360:])
                controller.set_baseline_gain(baseline)
            elif pre_gain_queue:
                baseline = sum(pre_gain_queue) / len(pre_gain_queue)
                controller.set_baseline_gain(baseline)
            else:
                controller.set_baseline_gain(controller.heater_gain)
        if isinstance(controller, AdaptiveController) and shift_step is not None and i >= shift_step:
            controller.update_normalized_gain()

        # Oracle：注入真实效率
        if isinstance(controller, OracleController):
            controller.set_true_efficiency(vg.plant.heater_efficiency)

    vg.stop()

    # 汇总分段
    seg_out = {}
    for name, s in segments.items():
        n = max(1, s["n"])
        seg_out[name] = {
            "mae_c": round(s["mae_sum"] / n, 4),
            "violation_degmin": round(s["violation_degmin"], 2),
            "energy_kwh": round(s["energy_kwh"], 3),
            "sat_ratio": round(s["sat_steps"] / n, 4),
            "travel_pct": round(s["travel_pct"], 1),
            "steps": s["n"],
        }

    out = {
        "controller": kind,
        "seed": seed,
        "segments": seg_out,
        "recovery_step": recovery_step,
        "recovery_minutes": round(recovery_step * dt / 60.0, 1) if recovery_step is not None else None,
        "true_heater_efficiency_trace": [round(x, 3) for x in true_eff_trace],
    }
    if isinstance(controller, AdaptiveController):
        out["est_gain_trace"] = [round(x, 4) for x in est_gain_trace]
        out["normalized_gain_trace"] = norm_gain_trace
        out["baseline_gain"] = controller.baseline_gain
        out["final_gain"] = round(controller.heater_gain, 4)
        out["update_count"] = controller.update_count
        out["unidentifiable_count"] = controller.unidentifiable_count
    return out


# ---------------------------------------------------------------------------
# 多 seed 实验
# ---------------------------------------------------------------------------

def run_experiment(config_path: Path, seeds: list[int],
                   output_path: Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    kinds = ["fixed_model", "adaptive_model", "oracle"]

    all_results: dict[str, Any] = {"scenario": cfg.scenario_id,
                                    "growth_stage": cfg.growth_stage,
                                    "seeds": seeds,
                                    "kinds": kinds,
                                    "runs": {}}
    for seed in seeds:
        all_results["runs"][str(seed)] = {}
        for kind in kinds:
            all_results["runs"][str(seed)][kind] = run_single(cfg, seed, kind)

    # 汇总：每个 kind 按 seed 平均
    summary = _summarize(all_results)
    all_results["summary"] = summary

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    return all_results


def _summarize(all_results) -> dict[str, Any]:
    seeds = all_results["seeds"]
    kinds = all_results["kinds"]
    runs = all_results["runs"]

    summary: dict[str, Any] = {}
    for kind in kinds:
        # 对每个分段指标按 seed 平均
        seg_names = ["pre", "0_1h", "1_3h", "3_6h", "6_12h"]
        agg: dict[str, Any] = {"segments": {}}
        for sn in seg_names:
            mae = [runs[str(s)][kind]["segments"][sn]["mae_c"] for s in seeds]
            viol = [runs[str(s)][kind]["segments"][sn]["violation_degmin"] for s in seeds]
            sat = [runs[str(s)][kind]["segments"][sn]["sat_ratio"] for s in seeds]
            energy = [runs[str(s)][kind]["segments"][sn]["energy_kwh"] for s in seeds]
            travel = [runs[str(s)][kind]["segments"][sn]["travel_pct"] for s in seeds]
            agg["segments"][sn] = {
                "mae_c_mean": round(sum(mae) / len(mae), 4),
                "violation_degmin_mean": round(sum(viol) / len(viol), 2),
                "sat_ratio_mean": round(sum(sat) / len(sat), 4),
                "energy_kwh_mean": round(sum(energy) / len(energy), 3),
                "travel_pct_mean": round(sum(travel) / len(travel), 1),
            }
        # 恢复时间（只对 adaptive 有意义，但统一记录）
        recov = [runs[str(s)][kind].get("recovery_minutes") for s in seeds]
        agg["recovery_minutes"] = [r for r in recov if r is not None]

        # 突变后全时段 = 0_1h+1_3h+3_6h+6_12h 加权平均
        post_mae = []
        post_viol = []
        for s in seeds:
            segs = runs[str(s)][kind]["segments"]
            wsum = sum(segs[sn]["steps"] for sn in ["0_1h", "1_3h", "3_6h", "6_12h"])
            post_mae.append(sum(segs[sn]["mae_c"] * segs[sn]["steps"] for sn in ["0_1h", "1_3h", "3_6h", "6_12h"]) / wsum)
            post_viol.append(sum(segs[sn]["violation_degmin"] for sn in ["0_1h", "1_3h", "3_6h", "6_12h"]))
        agg["post_shift_mae_c_mean"] = round(sum(post_mae) / len(post_mae), 4)
        agg["post_shift_violation_degmin_mean"] = round(sum(post_viol) / len(post_viol), 2)

        # 自适应专属：归一化增益收敛
        if kind == "adaptive_model":
            final_norm = [runs[str(s)][kind].get("final_gain", 0.0) / (runs[str(s)][kind].get("baseline_gain") or 1.0)
                          for s in seeds]
            agg["normalized_gain_final_mean"] = round(sum(final_norm) / len(final_norm), 4)

        summary[kind] = agg

    return summary
