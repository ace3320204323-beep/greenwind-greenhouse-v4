"""运行时适配器（任务书 8.2）。

让虚拟大棚对控制程序表现得像真实设备。控制器只能通过本适配器
读取观测与下发命令，不得直接引用 greenhouse.py 的内部变量。

接口：
    read_points() -> 统一测点列表
    read_latest() -> 最新观测值
    write_commands(command_batch) -> 提交控制命令
    read_feedback() -> 读取设备反馈
    health() -> 模拟设备与传感器健康状态
    stop() -> 停止仿真（不阻塞数据保存与资源释放）
"""

from __future__ import annotations

from typing import Any

from .runner import VirtualGreenhouse
from .schemas import ControlCommand, CommandFeedback


class VirtualGreenhouseAdapter:
    def __init__(self, greenhouse: VirtualGreenhouse):
        self._vg = greenhouse
        self._pending_feedback: list[CommandFeedback] = []
        self._last_obs: list = []

    # ------------------------------------------------------------------
    def read_points(self) -> list[str]:
        """返回统一测点 ID 列表。"""
        if self._vg._last_obs is None:
            return []
        seen: list[str] = []
        for o in self._vg._last_obs.observations:
            if o.point_id not in seen:
                seen.append(o.point_id)
        return seen

    def read_latest(self) -> dict[str, dict[str, Any]]:
        """返回最新观测值，key 为 point_id。"""
        if self._vg._last_obs is None:
            return {}
        return {o.point_id: o.to_dict() for o in self._vg._last_obs.observations}

    def write_commands(self, command_batch: list[ControlCommand]) -> None:
        """提交控制命令（在本周期内由 step 消费）。"""
        self._last_commands = command_batch

    def read_feedback(self) -> list[CommandFeedback]:
        """返回最近一次 step 的设备反馈。"""
        return self._pending_feedback

    def health(self) -> dict[str, Any]:
        """返回模拟设备与传感器健康状态。"""
        return {
            "greenhouse_id": self._vg.config.greenhouse_id,
            "sim_step": self._vg.sim_step,
            "sim_hours": self._vg.sim_seconds / 3600.0,
            "done": self._vg.sim_seconds / 3600.0 >= self._vg.duration_hours,
            "stopped": getattr(self._vg, "_stopped", False),
        }

    def stop(self) -> dict[str, Any]:
        """停止仿真并落盘。"""
        return self._vg.stop()
