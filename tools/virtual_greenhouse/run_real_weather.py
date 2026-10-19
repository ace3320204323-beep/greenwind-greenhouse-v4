"""真实天气 CSV 接入验证脚本。

用真实气象站 CSV 数据驱动虚拟大棚，跑一段连续实况天气验证（默认 7 天），
输出控制器在真实天气下的表现。

用法：
    python tools/virtual_greenhouse/run_real_weather.py [--days 7] [--csv <path>]

不改动 runner.py 主流程：通过把 VirtualGreenhouse.weather 替换为 WeatherCSVModel
实例来注入真实天气（两者 step() 接口一致）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from virtual_greenhouse.config import load_config
from virtual_greenhouse.runner import VirtualGreenhouse
from virtual_greenhouse.adapter import VirtualGreenhouseAdapter
from virtual_greenhouse.controllers import make_controller
from virtual_greenhouse.weather_csv import WeatherCSVModel

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "shandong_real_weather_14d.csv"


def run_real_weather(config_path: Path, csv_path: Path, days: int, seed: int,
                     controller_kind: str = "fixed_model"):
    cfg = load_config(config_path)
    day_min, day_max = cfg.targets.day_temp_min_c, cfg.targets.day_temp_max_c
    night_min, night_max = cfg.targets.night_temp_min_c, cfg.targets.night_temp_max_c
    target_setpoint = (day_min + day_max) / 2.0

    vg = VirtualGreenhouse(cfg, seed, evaluate=True)
    # 注入真实天气（替换合成天气，两者 step(sim_seconds) 接口一致）
    csv_model = WeatherCSVModel(csv_path, seed=seed,
                                meta={"source": "山东省气象局地面观测站",
                                      "location": "山东省济南市"})
    vg.weather = csv_model

    adapter = VirtualGreenhouseAdapter(vg)
    controller = make_controller(controller_kind, day_min, day_max,
                                 cfg.targets.humidity_max_pct,
                                 target=target_setpoint,
                                 night_temp_min=night_min, night_temp_max=night_max)

    vg.reset()
    adapter._last_obs = vg._last_obs

    steps = int(days * 24 * 3600 / cfg.simulation.dt_seconds)
    feedback_map: dict[str, float] = {}
    in_range = 0
    mae_sum = 0.0
    n = 0

    for i in range(steps):
        obs_map = adapter.read_latest()
        cmds = controller.decide(obs_map, feedback_map)
        result = vg.step(cmds)
        adapter._last_obs = result.observation
        feedback_map = {fb.device_id: fb.actual_value for fb in result.feedback}

        truth = result.evaluation_truth
        w = vg.weather.step(vg.sim_seconds)
        is_day = w.outdoor_solar_irradiance_w_m2 >= cfg.targets.day_radiation_threshold_w_m2
        tgt_min, tgt_max = (day_min, day_max) if is_day else (night_min, night_max)
        t = truth.zone_center_air_temperature_c
        if tgt_min <= t <= tgt_max:
            in_range += 1
            err = 0.0
        else:
            err = min(abs(t - tgt_min), abs(t - tgt_max))
        mae_sum += err
        n += 1

    print("=== 真实天气 CSV 接入验证结果 ===")
    for k, v in csv_model.summary().items():
        print(f"  {k}: {v}")
    print(f"  控制器: {controller_kind}")
    print(f"  仿真时长: {days} 天 ({steps} 步)")
    print(f"  温度达标率: {in_range / n:.3f}")
    print(f"  温度 MAE: {mae_sum / n:.2f}℃")
    return {"days": days, "controller": controller_kind,
            "in_range_ratio": round(in_range / n, 4),
            "mae_c": round(mae_sum / n, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--controller", type=str, default="fixed_model")
    ap.add_argument("--config", type=str,
                    default=str(Path(__file__).resolve().parent / "scenarios" / "shandong_winter_strawberry.json"))
    args = ap.parse_args()

    run_real_weather(Path(args.config), Path(args.csv), args.days,
                     args.seed, args.controller)
