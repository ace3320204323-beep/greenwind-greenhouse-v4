"""储能有/无对照实验。

同一天气、同一控制规则、同一随机种子下，比较“配置储能”和“无储能”两组。
默认在仿真第 10~14 小时安排市电停电，观察供电缺口、SOC 与棚温差异。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_greenhouse.config import load_config
from virtual_greenhouse.runner import VirtualGreenhouse
from virtual_greenhouse.schemas import (
    ActionType,
    CommandSource,
    ControlCommand,
    ControlMode,
    new_id,
)


DEVICES = {
    "heater_01": (ActionType.SET_POWER, "%"),
    "vent_01": (ActionType.SET_POSITION, "%"),
    "curtain_01": (ActionType.SET_POSITION, "%"),
    "fan_01": (ActionType.SET_POWER, "%"),
}


def observation_map(batch):
    return {o.point_id: o.value for o in batch.observations if o.value is not None}


def rule_targets(obs: dict[str, float]) -> dict[str, float]:
    """仅使用传感器观测的相同规则控制器，保证两组公平。"""
    temp = float(obs.get("center_air_temperature_c", 20.0))
    solar = float(obs.get("outdoor_solar_irradiance_w_m2", 0.0))
    is_day = solar >= 50.0
    low, high = (20.0, 25.0) if is_day else (8.0, 10.0)
    heater = 0.0
    vent = 0.0
    fan = 0.0
    curtain = 0.0 if is_day else 100.0
    if temp < low:
        heater = min(100.0, 35.0 + (low - temp) * 18.0)
    elif temp > high:
        vent = min(100.0, 25.0 + (temp - high) * 22.0)
        fan = min(100.0, max(0.0, (temp - high) * 20.0))
    return {
        "heater_01": heater,
        "vent_01": vent,
        "curtain_01": curtain,
        "fan_01": fan,
    }


def make_commands(targets: dict[str, float], sim_time: str, greenhouse_id: str):
    commands = []
    for device_id, target in targets.items():
        action, unit = DEVICES[device_id]
        commands.append(ControlCommand(
            command_id=new_id("energy-exp"),
            trace_id=new_id("trace"),
            greenhouse_id=greenhouse_id,
            device_id=device_id,
            action_type=action,
            target_value=float(target),
            unit=unit,
            issued_at=sim_time,
            effective_at=sim_time,
            expires_at=sim_time,
            control_mode=ControlMode.RULE,
            source=CommandSource.LOCAL_CONTROLLER,
            model_version="energy-storage-experiment-v1",
            safety_checked=True,
        ))
    return commands


def run_case(base_config, with_storage: bool, seed: int):
    cfg = copy.deepcopy(base_config)
    cfg.simulation.duration_hours = 24.0
    cfg.fault_schedule = []
    cfg.grid_schedule = [
        {"start_hour": 0.0, "end_hour": 10.0, "status": "NORMAL"},
        {"start_hour": 10.0, "end_hour": 14.0, "status": "OUTAGE"},
        {"start_hour": 14.0, "end_hour": 24.1, "status": "NORMAL"},
    ]
    if not with_storage:
        cfg.energy.max_charge_power_kw = 0.0
        cfg.energy.max_discharge_power_kw = 0.0
        cfg.energy.initial_soc_pct = cfg.energy.min_soc_pct

    greenhouse = VirtualGreenhouse(cfg, seed=seed, evaluate=True)
    batch = greenhouse.reset()
    rows = []
    for _ in range(int(24 * 3600 / cfg.simulation.dt_seconds)):
        targets = rule_targets(observation_map(batch))
        result = greenhouse.step(make_commands(targets, batch.sim_time, cfg.greenhouse_id))
        truth = result.evaluation_truth
        energy = result.energy["state"]
        next_obs = observation_map(result.observation)
        rows.append({
            "case": "with_storage" if with_storage else "without_storage",
            "sim_hour": round(greenhouse.sim_seconds / 3600.0, 4),
            "grid_status": energy["grid_status"],
            "center_temp_c": round(truth.zone_center_air_temperature_c, 4),
            "outdoor_temp_c": round(float(next_obs.get("outdoor_air_temperature_c", 0.0)), 4),
            "heater_requested_pct": round(targets["heater_01"], 3),
            "heater_effective_pct": round(result.energy["effective_actuators"].get("heater_01", 0.0), 3),
            "soc_pct": round(energy["battery_soc_pct"], 4),
            "battery_power_kw": round(energy["battery_power_kw"], 4),
            "renewable_power_kw": round(energy["renewable_power_kw"], 4),
            "grid_power_kw": round(energy["grid_power_kw"], 4),
            "load_power_kw": round(energy["load_power_kw"], 4),
            "unserved_power_kw": round(energy["unserved_power_kw"], 4),
            "power_supply_ratio": round(energy["power_supply_ratio"], 6),
        })
        batch = result.observation
    greenhouse.stop()
    return rows


def summarize(rows):
    outage = [r for r in rows if r["grid_status"] == "OUTAGE"]
    dt_h = 1.0 / 60.0
    return {
        "outage_min_temperature_c": round(min(r["center_temp_c"] for r in outage), 3),
        "outage_mean_supply_ratio": round(sum(r["power_supply_ratio"] for r in outage) / len(outage), 4),
        "unserved_energy_kwh": round(sum(r["unserved_power_kw"] * dt_h for r in rows), 3),
        "grid_energy_kwh": round(sum(r["grid_power_kw"] * dt_h for r in rows), 3),
        "renewable_generation_kwh": round(sum(r["renewable_power_kw"] * dt_h for r in rows), 3),
        "final_soc_pct": round(rows[-1]["soc_pct"], 3),
    }


def plot(rows_with, rows_without, output_path: Path):
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    x = [r["sim_hour"] for r in rows_with]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, [r["center_temp_c"] for r in rows_with], label="With storage", lw=2.2)
    axes[0].plot(x, [r["center_temp_c"] for r in rows_without], label="Without storage", lw=1.8, ls="--")
    axes[0].plot(x, [r["outdoor_temp_c"] for r in rows_with], label="Outdoor", lw=1.2, alpha=.7)
    axes[0].set_ylabel("Temperature (degC)")
    axes[0].set_title("Storage Coordination During Grid Outage (same weather and control rules)")
    axes[0].legend(ncol=3)

    axes[1].plot(x, [r["load_power_kw"] for r in rows_with], label="Load", color="#D97706")
    axes[1].plot(x, [r["renewable_power_kw"] for r in rows_with], label="Renewable", color="#16A34A")
    axes[1].plot(x, [r["grid_power_kw"] for r in rows_with], label="Grid", color="#2563EB")
    axes[1].plot(x, [r["battery_power_kw"] for r in rows_with], label="Battery (+discharge)", color="#7C3AED")
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(ncol=4)

    axes[2].plot(x, [r["soc_pct"] for r in rows_with], label="Battery SOC", color="#7C3AED", lw=2)
    axes[2].plot(x, [r["power_supply_ratio"] * 100 for r in rows_with], label="Supply ratio: storage", color="#16A34A")
    axes[2].plot(x, [r["power_supply_ratio"] * 100 for r in rows_without], label="Supply ratio: no storage", color="#DC2626", ls="--")
    axes[2].set_ylabel("Percent (%)")
    axes[2].set_xlabel("Simulation time (h)")
    axes[2].legend(ncol=3)

    for ax in axes:
        ax.axvspan(10, 14, color="#DC2626", alpha=.10, label="Grid outage")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=str(Path(__file__).parent / "scenarios" / "shandong_winter_strawberry.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "energy_results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.scenario)
    rows_with = run_case(config, True, args.seed)
    rows_without = run_case(config, False, args.seed)
    all_rows = rows_with + rows_without

    with (output_dir / "energy_storage_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "experiment": "storage_vs_no_storage",
        "seed": args.seed,
        "outage_window_hours": [10, 14],
        "with_storage": summarize(rows_with),
        "without_storage": summarize(rows_without),
        "boundary": "结果仅证明储能模型在当前仿真参数与规则下参与功率平衡，不代表真实项目收益。",
    }
    (output_dir / "energy_storage_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(rows_with, rows_without, output_dir / "储能协同对照实验.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"结果目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
