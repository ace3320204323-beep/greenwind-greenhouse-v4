"""热惯性阶跃响应实验。

目的：不以“代码里写了热容”作为证据，而是用可重复实验证明：
1. 加热开启后棚温逐步上升，不瞬时跳变；
2. 加热关闭后，冠层、基质和棚架蓄热体继续释热，使棚温平滑衰减；
3. 在相同天气和设备指令下，低热容对照组的变化更快，证明热容参数真正参与了计算。

运行：
    python tools/virtual_greenhouse/thermal_inertia_experiment.py

输出：
    tools/virtual_greenhouse/thermal_inertia_results/
      - thermal_inertia_trace.csv
      - thermal_inertia_metrics.json
      - thermal_inertia_step_response.png

说明：低热容组是敏感性对照，不代表另一座真实大棚。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from virtual_greenhouse.config import load_config
    from virtual_greenhouse.greenhouse import GreenhousePlant, ZONES
else:
    from .config import load_config
    from .greenhouse import GreenhousePlant, ZONES


def _constant_weather(outdoor_temp_c: float = 0.0):
    return SimpleNamespace(
        outdoor_air_temperature_c=outdoor_temp_c,
        outdoor_relative_humidity_pct=65.0,
        outdoor_solar_irradiance_w_m2=0.0,
        outdoor_wind_speed_m_s=1.0,
    )


def _initialise(plant: GreenhousePlant, temperature_c: float) -> None:
    for zid in ZONES:
        plant.state.air_temp[zid] = temperature_c
        plant.state.air_abs_humidity[zid] = 5.0
    plant.state.canopy_temp = temperature_c
    plant.state.substrate_temp = temperature_c
    plant.state.structure_temp = temperature_c
    plant.state.cover_temp = temperature_c


def _scaled_inertia_config(cfg, factor: float):
    scaled = copy.deepcopy(cfg)
    for zone in scaled.zones.values():
        zone.air_heat_capacity_kj_k *= factor
    scaled.plant.canopy_heat_capacity_kj_k *= factor
    scaled.plant.substrate_heat_capacity_kj_k *= factor
    scaled.plant.structure_heat_capacity_kj_k *= factor
    return scaled


def _run_case(cfg, label: str, total_hours: float, heater_on_hours: float,
              dt_seconds: float, seed: int) -> list[dict[str, float | str]]:
    plant = GreenhousePlant(cfg)
    _initialise(plant, 8.0)
    rng = random.Random(seed)
    weather = _constant_weather(0.0)
    rows: list[dict[str, float | str]] = []
    steps = int(total_hours * 3600 / dt_seconds)

    for step in range(steps + 1):
        hour = step * dt_seconds / 3600.0
        heater_pct = 100.0 if hour < heater_on_hours else 0.0
        actuators = {
            "heater_01": heater_pct,
            "vent_01": 0.0,
            "fan_01": 0.0,
            "curtain_01": 100.0,
            "irrigation_01": 0.0,
        }
        if step > 0:
            plant.step(step * dt_seconds, dt_seconds, weather, actuators, rng)
        rows.append({
            "case": label,
            "hour": round(hour, 6),
            "heater_pct": heater_pct,
            "outdoor_temp_c": weather.outdoor_air_temperature_c,
            "east_air_temp_c": plant.state.air_temp["EAST"],
            "center_air_temp_c": plant.state.air_temp["CENTER"],
            "west_air_temp_c": plant.state.air_temp["WEST"],
            "canopy_temp_c": plant.state.canopy_temp,
            "substrate_temp_c": plant.state.substrate_temp,
            "structure_temp_c": plant.state.structure_temp,
        })
    return rows


def _metrics(rows: list[dict[str, float | str]], heater_on_hours: float) -> dict[str, float]:
    temperatures = [float(r["center_air_temp_c"]) for r in rows]
    hours = [float(r["hour"]) for r in rows]
    off_idx = min(range(len(hours)), key=lambda i: abs(hours[i] - heater_on_hours))
    one_hour_idx = min(range(len(hours)), key=lambda i: abs(hours[i] - (heater_on_hours + 1.0)))
    deltas = [abs(b - a) for a, b in zip(temperatures, temperatures[1:])]

    start_t = temperatures[0]
    shutdown_t = temperatures[off_idx]
    threshold = start_t + 0.632 * (shutdown_t - start_t)
    tau_minutes = math.nan
    for hour, temp in zip(hours[:off_idx + 1], temperatures[:off_idx + 1]):
        if temp >= threshold:
            tau_minutes = hour * 60.0
            break

    return {
        "start_temperature_c": round(start_t, 3),
        "shutdown_temperature_c": round(shutdown_t, 3),
        "temperature_60min_after_shutdown_c": round(temperatures[one_hour_idx], 3),
        "cooling_drop_first_hour_c": round(shutdown_t - temperatures[one_hour_idx], 3),
        "max_one_step_change_c": round(max(deltas, default=0.0), 4),
        "time_to_63pct_rise_minutes": round(tau_minutes, 2),
    }


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, standard_rows, low_rows, heater_on_hours: float) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("绘图需要 matplotlib：pip install matplotlib") from exc

    # Windows 优先使用微软雅黑/黑体；无中文字体的运行环境自动回退英文，
    # 避免生成的验收图出现方框字符。
    cjk_font = None
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            cjk_font = candidate
            break
        except ValueError:
            continue
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    bilingual = cjk_font is not None
    label_standard = "Standard thermal mass / 标准热惯性" if bilingual else "Standard thermal mass"
    label_structure = "Structure thermal mass / 棚架蓄热体" if bilingual else "Structure thermal mass"
    label_substrate = "Substrate / 基质" if bilingual else "Substrate"
    label_low = "Low-capacity comparison / 低热容对照" if bilingual else "Low-capacity comparison"
    label_off = "Heater off / 停热" if bilingual else "Heater off"
    title = ("Greenhouse thermal-inertia step response / 虚拟大棚热惯性阶跃响应"
             if bilingual else "Greenhouse thermal-inertia step response")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    x = [float(r["hour"]) for r in standard_rows]
    ax1.plot(x, [r["center_air_temp_c"] for r in standard_rows],
             label=label_standard, linewidth=2.4)
    ax1.plot(x, [r["structure_temp_c"] for r in standard_rows],
             label=label_structure, linewidth=1.8)
    ax1.plot(x, [r["substrate_temp_c"] for r in standard_rows],
             label=label_substrate, linewidth=1.8)
    ax1.plot(x, [r["center_air_temp_c"] for r in low_rows],
             label=label_low, linestyle="--", linewidth=1.8)
    ax1.axvline(heater_on_hours, color="#d62728", linestyle=":", label=label_off)
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax1.legend(ncol=2, fontsize=9)

    ax2.step(x, [r["heater_pct"] for r in standard_rows], where="post", color="#d62728")
    ax2.set_xlabel("Simulation time (h)")
    ax2.set_ylabel("Heater (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="虚拟大棚热惯性阶跃响应实验")
    base = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(base / "scenarios" / "shandong_winter_strawberry.json"))
    parser.add_argument("--out", default=str(base / "thermal_inertia_results"))
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--heater-on-hours", type=float, default=3.0)
    parser.add_argument("--low-inertia-factor", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dt = cfg.simulation.dt_seconds
    standard_rows = _run_case(cfg, "standard", args.hours, args.heater_on_hours, dt, args.seed)
    low_cfg = _scaled_inertia_config(cfg, args.low_inertia_factor)
    low_rows = _run_case(low_cfg, "low_capacity_comparison", args.hours,
                         args.heater_on_hours, dt, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "thermal_inertia_trace.csv", standard_rows + low_rows)
    metrics = {
        "experiment": "thermal_inertia_step_response",
        "heater_on_hours": args.heater_on_hours,
        "low_inertia_factor": args.low_inertia_factor,
        "standard": _metrics(standard_rows, args.heater_on_hours),
        "low_capacity_comparison": _metrics(low_rows, args.heater_on_hours),
        "boundary": "本结果为仿真阶跃响应，热容参数仍需真实棚室数据标定。",
    }
    (out / "thermal_inertia_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(out / "thermal_inertia_step_response.png", standard_rows, low_rows,
          args.heater_on_hours)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"CSV: {out / 'thermal_inertia_trace.csv'}")
    print(f"PNG: {out / 'thermal_inertia_step_response.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
