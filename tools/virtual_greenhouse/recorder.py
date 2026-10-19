"""记录器与输出目录（任务书 13）。

每次运行建立独立目录 runs/{run_id}/，输出：
manifest.json, config_snapshot.json, telemetry.csv, truth.csv,
commands.csv, feedback.csv, events.csv, controller_trace.csv,
metrics.json, summary.txt
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schemas import json_safe


CSV_FIELDS = {
    "telemetry": ["sim_step", "point_id", "greenhouse_id", "zone_id", "metric",
                  "value", "unit", "observed_at", "quality", "sequence"],
    "truth": ["sim_step", "sim_time", "zone_east_air_temperature_c",
              "zone_center_air_temperature_c", "zone_west_air_temperature_c",
              "zone_east_absolute_humidity_g_m3", "zone_center_absolute_humidity_g_m3",
              "zone_west_absolute_humidity_g_m3", "canopy_temperature_c",
              "substrate_temperature_c", "substrate_moisture_pct", "cover_temperature_c",
              "indoor_co2_ppm", "actual_vent_position_pct", "actual_fan_power_pct",
              "actual_curtain_position_pct", "actual_heater_power_pct",
              "true_heat_transfer_coefficient", "true_ventilation_coefficient",
              "true_solar_gain_coefficient", "true_air_leakage_rate"],
    "commands": ["command_id", "trace_id", "device_id", "action_type", "target_value",
                 "unit", "control_mode", "source", "priority", "safety_checked", "issued_at"],
    "feedback": ["feedback_id", "command_id", "trace_id", "device_id", "status",
                 "target_value", "actual_value", "unit", "fault_code", "message"],
    "events": ["event_id", "event_type", "sim_time", "sim_step", "detail"],
    "controller_trace": ["trace_id", "sim_step", "model_name", "model_version",
                         "prediction_horizon_steps", "predicted_temperature_c",
                         "prediction_confidence", "objective_total_cost",
                         "safety_check_result"],
}


class Recorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, csv.DictWriter] = {}
        self._files: dict[str, Any] = {}
        for name, fields in CSV_FIELDS.items():
            f = open(run_dir / f"{name}.csv", "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            self._writers[name] = w
            self._files[name] = f

    def write(self, name: str, row: dict[str, Any]) -> None:
        if name in self._writers:
            self._writers[name].writerow({k: json_safe(row.get(k, "")) for k in CSV_FIELDS[name]})

    def close(self) -> None:
        for f in self._files.values():
            f.flush()

    def finalize(self, manifest: dict, config_snapshot: dict, metrics: dict,
                 summary: str) -> None:
        self._files["telemetry"].close()
        self._files["truth"].close()
        self._files["commands"].close()
        self._files["feedback"].close()
        self._files["events"].close()
        self._files["controller_trace"].close()

        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(json_safe(config_snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "metrics.json").write_text(
            json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "summary.txt").write_text(summary, encoding="utf-8")


def build_summary(manifest: dict, metrics: dict, event_count: int) -> str:
    env = metrics.get("environment", {})
    ctrl = metrics.get("control", {})
    safety = metrics.get("safety", {})
    lines = [
        "绿风润耕虚拟大棚仿真摘要",
        "=" * 40,
        f"run_id       : {manifest.get('run_id')}",
        f"scenario_id  : {manifest.get('scenario_id')}",
        f"seed         : {manifest.get('seed')}",
        f"duration_h   : {manifest.get('duration_hours')}",
        f"events       : {event_count}",
        "",
        "环境指标",
        f"  温度达标率   : {env.get('temperature_in_range_ratio')}",
        f"  温度 MAE    : {env.get('temperature_mae_c')} ℃",
        f"  温度 RMSE   : {env.get('temperature_rmse_c')} ℃",
        f"  越界度·分   : {env.get('temperature_violation_degree_minutes')}",
        f"  湿度达标率   : {env.get('humidity_in_range_ratio')}",
        f"  三区最大温差 : {env.get('zone_temperature_max_difference_c')} ℃",
        "",
        "控制指标",
        f"  指令数      : {ctrl.get('command_count')}",
        f"  执行器行程  : {ctrl.get('actuator_travel_total_pct')} %",
        f"  参数突变恢复: {ctrl.get('recovery_time_after_shift_min')} min",
        "",
        "安全指标",
        f"  拒绝无效命令: {safety.get('invalid_command_rejected_count')}",
        f"  不安全动作  : {safety.get('unsafe_action_count')} (须为 0)",
        f"  传感器缺失率: {safety.get('sensor_missing_ratio')}",
        "",
        "边界声明：本结果属于带模型失配与故障注入条件下的仿真验证，",
        "不等同于真实大棚现场效果。",
    ]
    return "\n".join(lines)
