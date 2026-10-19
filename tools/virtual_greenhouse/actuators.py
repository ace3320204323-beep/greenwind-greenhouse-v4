"""执行器模型。

接收控制命令，计算设备实际开度/功率。核心原则（任务书 10.6 / 18.4）：
- 命令目标值 != 设备实际值。
- 实际值按最大动作速度、死区、回差、最小启停时间、延迟与故障状态演化。
- actual(t+1) = actual(t) + clip(command(t) - actual(t), -rate*dt, rate*dt)

死区与回差：目标落在 [actual - deadband, actual + deadband] 内不动作；
改变方向时需越过回差带宽才反向，避免抖动。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DeviceSpec
from .schemas import DeviceState, FeedbackStatus, CommandFeedback, new_id


@dataclass
class ActuatorState:
    spec: DeviceSpec
    actual: float = 0.0
    target: float = 0.0
    state: DeviceState = DeviceState.OFF
    fault_code: str = ""
    # 机械保护相关
    last_start_sim_s: float = -1e18
    last_stop_sim_s: float = -1e18
    moving_direction: int = 0       # 1 打开方向，-1 关闭方向，0 静止
    stuck: bool = False             # 卡涩
    failed: bool = False            # 完全失效
    max_position_limit_pct: float = 100.0  # 卡涩后最大可达位置


class ActuatorModel:
    """管理所有设备的实际状态演化。"""

    def __init__(self, specs: list[DeviceSpec]):
        self.devices: dict[str, ActuatorState] = {
            s.device_id: ActuatorState(spec=s) for s in specs
        }

    def apply_command(self, device_id: str, target: float, sim_seconds: float,
                      dt_seconds: float) -> CommandFeedback:
        """写入目标值并返回即时反馈。实际值在 step 中演化。"""
        st = self.devices.get(device_id)
        if st is None:
            return CommandFeedback(
                feedback_id=new_id("fb"), command_id="", trace_id="",
                device_id=device_id, status=FeedbackStatus.REJECTED,
                target_value=target, actual_value=0.0, unit="%",
                feedback_at="", fault_code="UNKNOWN_DEVICE",
                message="device not found",
            )
        if st.failed:
            st.target = target  # 记录但无法执行
            return CommandFeedback(
                feedback_id=new_id("fb"), command_id="", trace_id="",
                device_id=device_id, status=FeedbackStatus.FAILED,
                target_value=target, actual_value=st.actual, unit=st.spec.unit,
                feedback_at="", fault_code=st.fault_code,
                message="device faulted",
            )
        # 范围裁剪
        target = max(0.0, min(st.max_position_limit_pct, target))
        st.target = target
        return CommandFeedback(
            feedback_id=new_id("fb"), command_id="", trace_id="",
            device_id=device_id, status=FeedbackStatus.ACCEPTED,
            target_value=target, actual_value=st.actual, unit=st.spec.unit,
            feedback_at="", message="accepted",
        )

    def step(self, sim_seconds: float, dt_seconds: float) -> None:
        """演化所有设备的实际值。"""
        for st in self.devices.values():
            self._step_one(st, sim_seconds, dt_seconds)

    def _step_one(self, st: ActuatorState, sim_seconds: float, dt_seconds: float) -> None:
        spec = st.spec
        if st.failed:
            st.actual = 0.0
            st.state = DeviceState.FAULT
            return

        error = st.target - st.actual
        direction = 1 if error > 0 else -1 if error < 0 else 0

        # 死区：误差小于死区不动作
        if abs(error) <= spec.deadband:
            st.state = DeviceState.OFF if st.actual <= spec.deadband else DeviceState.RUNNING
            return

        # 回差：换向需要超过回差带宽
        if direction != 0 and st.moving_direction != 0 and direction != st.moving_direction:
            if abs(error) <= spec.hysteresis:
                return

        # 最小启停时间约束
        if direction > 0 and sim_seconds - st.last_stop_sim_s < spec.min_stop_seconds:
            return
        if direction < 0 and sim_seconds - st.last_start_sim_s < spec.min_run_seconds:
            return

        # 卡涩：行程只能达到 max_position_limit
        if st.stuck:
            st.actual = min(st.actual, st.max_position_limit_pct)
            if direction > 0 and st.actual >= st.max_position_limit_pct:
                st.state = DeviceState.FAULT
                return

        # 行程速率积分
        step = spec.max_rate_per_s * dt_seconds
        new_actual = st.actual + max(-step, min(step, error))
        new_actual = max(0.0, min(st.max_position_limit_pct, new_actual))

        # 记录启停时刻
        if st.actual <= spec.deadband and new_actual > spec.deadband:
            st.last_start_sim_s = sim_seconds
        if st.actual > spec.deadband and new_actual <= spec.deadband:
            st.last_stop_sim_s = sim_seconds

        st.actual = new_actual
        st.moving_direction = direction
        st.state = (DeviceState.RUNNING if st.actual > spec.deadband
                    else DeviceState.OFF)

    def snapshot(self) -> dict[str, ActuatorState]:
        return dict(self.devices)

    # 故障注入
    def set_stuck(self, device_id: str, max_position_pct: float) -> None:
        st = self.devices.get(device_id)
        if st:
            st.stuck = True
            st.max_position_limit_pct = min(st.max_position_limit_pct, max_position_pct)
            st.fault_code = "STUCK"

    def set_failed(self, device_id: str) -> None:
        st = self.devices.get(device_id)
        if st:
            st.failed = True
            st.fault_code = "FAILED"
            st.state = DeviceState.FAULT
