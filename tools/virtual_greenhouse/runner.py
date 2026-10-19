"""虚拟大棚主接口（任务书 8.1 / 9）。

VirtualGreenhouse.reset(config, seed) -> ObservationBatch
VirtualGreenhouse.step(command_batch) -> StepResult
VirtualGreenhouse.stop() -> None   （幂等，可重复调用）

严格按第十三章的执行顺序组织每一步。真值（truth）仅在评价模式返回，
控制器不得读取未来天气与真实参数。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import GreenhouseConfig, load_config
from .weather import WeatherModel
from .greenhouse import GreenhousePlant
from .actuators import ActuatorModel
from .sensors import SensorModel
from .faults import FaultScheduler
from .metrics import MetricsAccumulator
from .energy_system import EnergySystem
from .schemas import (
    CN_TZ, Observation, ObservationBatch, ControlCommand, CommandFeedback,
    Event, FeedbackStatus, CommandSource, ControlMode,
    new_id, json_safe,
)


@dataclass
class StepResult:
    observation: ObservationBatch
    feedback: list[CommandFeedback]
    events: list[Event]
    done: bool
    metrics_partial: dict[str, Any]
    evaluation_truth: Any = None   # 仅评价模式填充
    energy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "feedback": [f.to_dict() for f in self.feedback],
            "events": [e.to_dict() for e in self.events],
            "done": self.done,
            "metrics_partial": json_safe(self.metrics_partial),
            "evaluation_truth": json_safe(self.evaluation_truth.to_dict())
            if self.evaluation_truth is not None else None,
            "energy": json_safe(self.energy),
        }


class VirtualGreenhouse:
    def __init__(self, config: GreenhouseConfig, seed: int,
                 evaluate: bool = False, output_dir: str | Path | None = None):
        self.config = config
        self.seed = seed
        self.evaluate = evaluate
        self.rng = random.Random(seed)

        self.dt = config.simulation.dt_seconds
        self.duration_hours = config.simulation.duration_hours
        self.sim_seconds = 0.0
        self.sim_step = 0
        start = datetime.fromisoformat(config.simulation.sim_start)
        self.start_dt = start
        hour_offset = start.hour + start.minute / 60.0 + start.second / 3600.0
        self._last_obs: ObservationBatch | None = None

        self.weather = WeatherModel(config.weather, seed, config.weather_schedule,
                                    hour_offset=hour_offset)
        self.plant = GreenhousePlant(config)
        self.actuators = ActuatorModel(config.devices)
        self.energy = EnergySystem(
            config.energy,
            config.devices,
            enabled=config.energy_enabled,
            grid_schedule=config.grid_schedule,
        )
        self.sensors = SensorModel(config.sensor, seed)
        self.faults = FaultScheduler(config.fault_schedule)
        self.metrics = MetricsAccumulator()

        self.run_id = new_id("run")
        self.sequence = 0

        from .recorder import Recorder
        self.recorder = Recorder(Path(output_dir)) if output_dir else None
        self._event_count = 0

        self._heater_kwh = 0.0
        self._fan_kwh = 0.0

        self._stopped = False
        self._finalized = False

    # ------------------------------------------------------------------
    def _sim_time_str(self) -> str:
        return (self.start_dt + timedelta(seconds=self.sim_seconds)).isoformat(timespec="seconds")

    def reset(self) -> ObservationBatch:
        w = self.weather.step(0.0)
        truth = self.plant.snapshot(self._sim_time_str(), 0, self.dt,
                                    self._actuator_map())
        self.energy.apply_to_truth(truth)
        obs = self.sensors.observe(truth, w, self._sim_time_str(), 0, 0.0, self.dt,
                                   self.run_id, self.config.scenario_id, self.seed,
                                   self.config.greenhouse_id, self.sequence)
        self._last_obs = ObservationBatch(observations=obs, sim_time=self._sim_time_str(),
                                          sim_step=0)
        return self._last_obs

    def _actuator_map(self) -> dict[str, float]:
        return {did: st.actual for did, st in self.actuators.devices.items()}

    # ------------------------------------------------------------------
    def step(self, command_batch: list[ControlCommand] | None = None) -> StepResult:
        command_batch = command_batch or []

        sim_time = self._sim_time_str()
        self.sim_step += 1
        self.sim_seconds += self.dt
        sim_hours = self.sim_seconds / 3600.0

        w = self.weather.step(self.sim_seconds)

        feedbacks: list[CommandFeedback] = []
        accepted: dict[str, float] = {}
        for cmd in command_batch:
            if not isinstance(cmd, ControlCommand):
                continue
            if cmd.target_value < 0 or cmd.target_value > 100:
                feedbacks.append(CommandFeedback(
                    feedback_id=new_id("fb"), command_id=cmd.command_id,
                    trace_id=cmd.trace_id, device_id=cmd.device_id,
                    status=FeedbackStatus.REJECTED, target_value=cmd.target_value,
                    actual_value=0.0, unit=cmd.unit, feedback_at=sim_time,
                    fault_code="OUT_OF_RANGE", message="target out of range",
                ))
                self.metrics.invalid_command_rejected += 1
                continue
            fb = self.actuators.apply_command(cmd.device_id, cmd.target_value,
                                              self.sim_seconds, self.dt)
            fb.command_id = cmd.command_id
            fb.trace_id = cmd.trace_id
            fb.feedback_at = sim_time
            feedbacks.append(fb)
            accepted[cmd.device_id] = cmd.target_value
            self.metrics.command_count += 1
            self.metrics.command_issued += 1
            if fb.status in (FeedbackStatus.RECEIVED, FeedbackStatus.ACCEPTED,
                             FeedbackStatus.EXECUTING, FeedbackStatus.SUCCEEDED):
                self.metrics.command_feedback_match += 1
            if self.recorder:
                self.recorder.write("commands", {
                    "command_id": cmd.command_id, "trace_id": cmd.trace_id,
                    "device_id": cmd.device_id, "action_type": cmd.action_type.value,
                    "target_value": cmd.target_value, "unit": cmd.unit,
                    "control_mode": cmd.control_mode.value, "source": cmd.source.value,
                    "priority": cmd.priority, "safety_checked": cmd.safety_checked,
                    "issued_at": cmd.issued_at,
                })

        # 5. 设备实际值演化
        self.actuators.step(self.sim_seconds, self.dt)

        # 5b. 反馈反映设备演化后的实际值（真实设备反馈，控制器可读）
        actual_map = self._actuator_map()
        for fb in feedbacks:
            if fb.status not in (FeedbackStatus.REJECTED, FeedbackStatus.FAILED):
                fb.actual_value = actual_map.get(fb.device_id, fb.actual_value)

        # 6. 能源平衡：设备实际状态形成负荷，市电/新能源/储能决定有效供电。
        energy_result = self.energy.step(
            self.sim_seconds,
            self.dt,
            w,
            actual_map,
            actuator_states=self.actuators.devices,
        )

        # 7. 物理计算 + 积分。加热/风机等使用能源子系统返回的实际可供功率。
        actuators_map = actual_map
        effective_actuators = energy_result.effective_actuators
        self.plant.step(self.sim_seconds, self.dt, w, effective_actuators, self.rng)

        # 9. 生成真实状态
        truth = self.plant.snapshot(sim_time, self.sim_step, self.dt, actuators_map)
        self.energy.apply_to_truth(truth)

        # 10. 传感器观测
        self.sequence += 1
        obs = self.sensors.observe(truth, w, sim_time, self.sim_step, self.sim_seconds,
                                   self.dt, self.run_id, self.config.scenario_id,
                                   self.seed, self.config.greenhouse_id, self.sequence)

        # 11. 故障注入
        events = self.faults.apply_due(sim_hours, self.sim_seconds, self.sim_step,
                                       sim_time, self.sensors, self.actuators, self.plant)

        # 12. 记录
        if self.recorder:
            for o in obs:
                self.recorder.write("telemetry", {
                    "sim_step": self.sim_step, "point_id": o.point_id,
                    "greenhouse_id": o.greenhouse_id, "zone_id": o.zone_id,
                    "metric": o.metric, "value": o.value, "unit": o.unit,
                    "observed_at": o.observed_at, "quality": o.quality.value,
                    "sequence": o.sequence,
                })
            self.recorder.write("truth", truth.to_dict())
            for fb in feedbacks:
                self.recorder.write("feedback", fb.to_dict())
            for ev in events:
                self.recorder.write("events", ev.to_dict())
        self._event_count += len(events)

        # 指标累计：昼夜目标
        is_day = w.outdoor_solar_irradiance_w_m2 >= self.config.targets.day_radiation_threshold_w_m2
        if is_day:
            tgt_min = self.config.targets.day_temp_min_c
            tgt_max = self.config.targets.day_temp_max_c
        else:
            tgt_min = self.config.targets.night_temp_min_c
            tgt_max = self.config.targets.night_temp_max_c
        self.metrics.record_step(truth, tgt_min, tgt_max,
                                 self.config.targets.humidity_min_pct,
                                 self.config.targets.humidity_max_pct,
                                 obs, self.dt, actuators_map)

        self._heater_kwh += effective_actuators.get("heater_01", 0.0) / 100.0 * 30.0 * self.dt / 3600.0
        self._fan_kwh += effective_actuators.get("fan_01", 0.0) / 100.0 * 1.5 * self.dt / 3600.0

        done = sim_hours >= self.duration_hours

        batch = ObservationBatch(observations=obs, sim_time=sim_time,
                                 sim_step=self.sim_step)
        self._last_obs = batch

        result = StepResult(
            observation=batch, feedback=feedbacks, events=events, done=done,
            metrics_partial={
                **self.metrics.summarize(self._heater_kwh, self._fan_kwh),
                "energy": self.energy.summary(),
            },
            evaluation_truth=truth if self.evaluate else None,
            energy=energy_result.to_dict(),
        )
        return result

    # ------------------------------------------------------------------
    def stop(self) -> dict[str, Any]:
        if self._stopped:
            return self._summary
        self._stopped = True
        self._summary = self.finalize()
        return self._summary

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return self._summary
        metrics = self.metrics.summarize(self._heater_kwh, self._fan_kwh)
        metrics["energy"] = self.energy.summary()
        manifest = {
            "run_id": self.run_id,
            "scenario_id": self.config.scenario_id,
            "seed": self.seed,
            "greenhouse_id": self.config.greenhouse_id,
            "crop_type": self.config.crop_type,
            "crop_stage": self.config.crop_stage.value,
            "software_version": "virtual-greenhouse-v4.2-energy",
            "duration_hours": self.duration_hours,
            "dt_seconds": self.dt,
            "sim_steps": self.sim_step,
            "started_at": self.start_dt.isoformat(timespec="seconds"),
            "ended_at": self._sim_time_str(),
            "config_fingerprint": _cfg_fp(self.config),
        }
        if self.recorder:
            from .recorder import build_summary
            summary = build_summary(manifest, metrics, self._event_count)
            self.recorder.finalize(manifest, self.config.to_dict(), metrics, summary)
        self._finalized = True
        self._summary = {"manifest": manifest, "metrics": metrics}
        return self._summary


def _cfg_fp(cfg: GreenhouseConfig) -> str:
    import hashlib
    return hashlib.sha256(
        json_dumps_safe(cfg.to_dict()).encode("utf-8")).hexdigest()[:16]


def json_dumps_safe(obj: Any) -> str:
    import json
    return json.dumps(json_safe(obj), sort_keys=True, ensure_ascii=False, default=str)
