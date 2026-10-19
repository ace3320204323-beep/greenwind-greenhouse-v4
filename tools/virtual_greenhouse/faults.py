"""故障与参数突变调度器（任务书 12）。

按时间计划注入故障与参数突变，并支持随机故障。故障事件写入 events 列表，
供 Recorder 落盘。故障类型：
- 传感器卡死 sensor_stuck、跳变 sensor_spike、漂移 sensor_drift、丢包 packet_loss、断连 sensor_disconnect
- 风口卡涩 vent_stuck、风机失效 fan_failed、保温帘卡涩 curtain_stuck、加热能力下降 heater_degraded
- 通信中断 comm_break
- 棚体参数突变 parameter_shift
"""

from __future__ import annotations

from .schemas import Event, new_id


class FaultScheduler:
    def __init__(self, schedule: list[dict]):
        self.schedule = sorted(schedule, key=lambda x: x.get("start_hour", 0.0))
        self._cursor = 0
        self._active_packet_loss = 0.0
        self._comm_broken = False

    def apply_due(self, sim_hours: float, sim_seconds: float, sim_step: int,
                  sim_time: str, sensors, actuators, plant) -> list[Event]:
        """把到达触发时刻的故障注入到对应模块，返回事件列表。"""
        events: list[Event] = []
        while self._cursor < len(self.schedule):
            item = self.schedule[self._cursor]
            if item.get("start_hour", 0.0) > sim_hours:
                break
            self._cursor += 1
            etype = item.get("event_type", "")
            detail = dict(item.get("detail", {}))
            event = self._dispatch(etype, detail, sim_seconds, sim_step, sim_time,
                                   sensors, actuators, plant)
            if event is not None:
                events.append(event)
        return events

    def _dispatch(self, etype: str, detail: dict, sim_seconds: float, sim_step: int,
                  sim_time: str, sensors, actuators, plant) -> Event | None:
        if etype == "sensor_stuck":
            pid = detail.get("point_id", "")
            sensors.inject_stuck(pid, 0.0)
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "sensor_spike":
            pid = detail.get("point_id", "")
            sensors.inject_spike(pid)
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "sensor_drift":
            pid = detail.get("point_id", "")
            sensors.inject_drift(pid, detail.get("rate_per_hour", 0.5))
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "packet_loss":
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "sensor_disconnect":
            pid = detail.get("point_id", "")
            sensors.inject_disconnect(pid)
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "vent_stuck":
            did = detail.get("device_id", "vent_01")
            actuators.set_stuck(did, detail.get("max_position_pct", 50.0))
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "fan_failed":
            did = detail.get("device_id", "fan_01")
            actuators.set_failed(did)
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "curtain_stuck":
            did = detail.get("device_id", "curtain_01")
            actuators.set_stuck(did, detail.get("max_position_pct", 40.0))
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "heater_degraded":
            # 加热能力下降：设置加热效率（真实参数，控制器不读取）
            eff = detail.get("efficiency", 0.75)
            plant.set_heater_efficiency(eff)
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "parameter_shift":
            plant.apply_parameter_shift(detail.get("parameter", ""),
                                        detail.get("factor", 1.0))
            return self._ev(etype, sim_time, sim_step, detail)
        if etype == "comm_break":
            self._comm_broken = True
            return self._ev(etype, sim_time, sim_step, detail)
        return None

    @staticmethod
    def _ev(etype: str, sim_time: str, sim_step: int, detail: dict) -> Event:
        return Event(event_id=new_id("ev"), event_type=etype, sim_time=sim_time,
                     sim_step=sim_step, detail=detail)
