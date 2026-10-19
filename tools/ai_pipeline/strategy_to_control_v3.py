"""绿风润耕 SAC V3 上层策略到设备控制的约束滚动优化器。

SAC 不再直接输出设备开度，而是输出四个策略量：

1. 温度目标修正（-2~2 摄氏度）；
2. 环境舒适权重（0~1）；
3. 节能权重（0~1）；
4. 排湿优先级（0~1）。

本模块使用 Transformer 的 10/30 分钟温度预测作为基线，对有限组候选设备动作
进行滚动代价评估，并在输出前强制执行互斥、变化率和最小保持时间约束。

说明：这是用于仿真研发与接口验证的轻量级离散滚动优化器。接入真实大棚后，
需要使用现场辨识参数重新标定动作响应模型和约束，不应把当前系数直接视为真实
设备参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEVICE_NAMES = ("heater", "vent", "curtain", "fan")
STRATEGY_NAMES = (
    "temperature_offset_c",
    "comfort_weight",
    "energy_weight",
    "dehumidify_priority",
)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _distance_to_band(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


@dataclass(frozen=True)
class StrategyAction:
    """SAC 输出经物理范围映射后的上层策略。"""

    temperature_offset_c: float
    comfort_weight: float
    energy_weight: float
    dehumidify_priority: float

    @classmethod
    def from_normalized(cls, action: Sequence[float]) -> "StrategyAction":
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape != (4,):
            raise ValueError("V3 策略动作必须为 4 维")
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        raw = np.clip(raw, -1.0, 1.0)
        return cls(
            temperature_offset_c=float(raw[0] * 2.0),
            comfort_weight=float((raw[1] + 1.0) * 0.5),
            energy_weight=float((raw[2] + 1.0) * 0.5),
            dehumidify_priority=float((raw[3] + 1.0) * 0.5),
        )

    def normalized(self) -> np.ndarray:
        return np.asarray(
            [
                self.temperature_offset_c / 2.0,
                self.comfort_weight * 2.0 - 1.0,
                self.energy_weight * 2.0 - 1.0,
                self.dehumidify_priority * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ControllerState:
    """滚动优化所需的当前状态与预测状态。"""

    temperature_c: float
    relative_humidity_pct: float
    outdoor_temperature_c: float
    outdoor_humidity_pct: float
    solar_w_m2: float
    predicted_10min_c: float
    predicted_30min_c: float
    target_low_c: float
    target_high_c: float
    is_day: bool


@dataclass(frozen=True)
class PlanResult:
    strategy: StrategyAction
    command_pct: Dict[str, float]
    unconstrained_command_pct: Dict[str, float]
    objective: float
    predicted_temperature_10min_c: float
    predicted_temperature_30min_c: float
    predicted_humidity_30min_pct: float
    constraint_events: Tuple[str, ...]


class StrategyToControlV3:
    """把上层策略转换为满足设备约束的控制指令。"""

    MAX_DELTA = {"heater": 25.0, "vent": 20.0, "curtain": 15.0, "fan": 25.0}

    def __init__(self, minimum_hold_steps: int = 2) -> None:
        self.minimum_hold_steps = max(0, int(minimum_hold_steps))
        self._step = 0
        self._last_switch_step = {name: -10_000 for name in DEVICE_NAMES}
        self._previous_command = {name: 0.0 for name in DEVICE_NAMES}

    def reset(self, previous_command: Mapping[str, float] | None = None) -> None:
        self._step = 0
        self._last_switch_step = {name: -10_000 for name in DEVICE_NAMES}
        previous_command = previous_command or {}
        self._previous_command = {
            name: _clip(float(previous_command.get(name, 0.0)), 0.0, 100.0)
            for name in DEVICE_NAMES
        }

    @staticmethod
    def _active(value: float) -> bool:
        return value > 5.0

    def _candidate_commands(self, previous: Mapping[str, float]) -> Iterable[Dict[str, float]]:
        """生成互斥的有限候选集；候选集中不会同时强加热和强通风。"""

        thermal_modes = [(0.0, 0.0, 0.0)]
        thermal_modes.extend((level, 0.0, 0.0) for level in (20.0, 40.0, 60.0, 80.0, 100.0))
        thermal_modes.extend((0.0, level, level) for level in (20.0, 40.0, 60.0, 80.0, 100.0))
        thermal_modes.extend((0.0, 0.0, level) for level in (25.0, 50.0, 75.0))

        curtain_levels = {
            0.0,
            50.0,
            100.0,
            round(float(previous.get("curtain", 0.0)), 3),
        }
        for (heater, vent, fan), curtain in product(thermal_modes, sorted(curtain_levels)):
            yield {
                "heater": heater,
                "vent": vent,
                "curtain": curtain,
                "fan": fan,
            }

    @staticmethod
    def _predict_candidate(
        state: ControllerState,
        command: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> Tuple[float, float, float]:
        """在 Transformer 基线预测上叠加候选动作的近似增量响应。"""

        heater_delta = (command["heater"] - previous.get("heater", 0.0)) / 100.0
        vent_delta = (command["vent"] - previous.get("vent", 0.0)) / 100.0
        fan_delta = (command["fan"] - previous.get("fan", 0.0)) / 100.0
        curtain_delta = (command["curtain"] - previous.get("curtain", 0.0)) / 100.0

        def project(base: float, horizon_factor: float) -> float:
            heating = 2.4 * horizon_factor * heater_delta
            mixing = (
                0.18
                * horizon_factor
                * vent_delta
                * (state.outdoor_temperature_c - base)
            )
            forced_mixing = (
                0.06
                * horizon_factor
                * max(0.0, fan_delta)
                * (state.outdoor_temperature_c - base)
            )
            if state.is_day:
                curtain_effect = (
                    -1.2
                    * horizon_factor
                    * curtain_delta
                    * _clip(state.solar_w_m2 / 1000.0, 0.0, 1.2)
                )
            else:
                curtain_effect = 0.8 * horizon_factor * curtain_delta
            return base + heating + mixing + forced_mixing + curtain_effect

        pred10 = project(state.predicted_10min_c, 1.0 / 3.0)
        pred30 = project(state.predicted_30min_c, 1.0)
        air_exchange = max(command["vent"], 0.6 * command["fan"]) / 100.0
        pred_rh30 = (
            state.relative_humidity_pct
            + 0.25 * air_exchange * (state.outdoor_humidity_pct - state.relative_humidity_pct)
            - 5.0 * command["heater"] / 100.0
            - 4.0 * command["fan"] / 100.0
        )
        return pred10, pred30, _clip(pred_rh30, 0.0, 100.0)

    @staticmethod
    def _objective(
        state: ControllerState,
        strategy: StrategyAction,
        command: Mapping[str, float],
        previous: Mapping[str, float],
        pred10: float,
        pred30: float,
        pred_rh30: float,
    ) -> float:
        low = state.target_low_c + strategy.temperature_offset_c
        high = state.target_high_c + strategy.temperature_offset_c
        comfort_weight = 0.5 + 2.5 * strategy.comfort_weight
        energy_weight = 0.15 + 1.5 * strategy.energy_weight
        humidity_weight = 0.15 + 1.8 * strategy.dehumidify_priority
        movement_weight = 0.10 + 0.35 * strategy.energy_weight

        temperature_cost = (
            _distance_to_band(pred10, low, high) ** 2
            + 1.5 * _distance_to_band(pred30, low, high) ** 2
        )
        humidity_cost = _distance_to_band(pred_rh30, 60.0, 85.0) ** 2 / 100.0
        energy = (
            command["heater"] / 100.0
            + 0.10 * command["fan"] / 100.0
            + 0.05 * command["vent"] / 100.0
            + 0.02 * command["curtain"] / 100.0
        )
        movement = float(
            np.mean(
                [
                    abs(command[name] - float(previous.get(name, 0.0))) / 100.0
                    for name in DEVICE_NAMES
                ]
            )
        )

        # 高湿时只有排湿优先级足够高，才值得承担通风带来的温度代价。
        humidity_urgency = max(0.0, state.relative_humidity_pct - 85.0) / 15.0
        insufficient_air_exchange = max(
            0.0, humidity_urgency * strategy.dehumidify_priority - command["vent"] / 100.0
        )
        return float(
            comfort_weight * temperature_cost
            + humidity_weight * humidity_cost
            + energy_weight * energy
            + movement_weight * movement
            + 1.5 * insufficient_air_exchange**2
        )

    def _apply_constraints(
        self,
        desired: Mapping[str, float],
        previous: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Tuple[str, ...]]:
        constrained = {
            name: _clip(float(desired.get(name, 0.0)), 0.0, 100.0)
            for name in DEVICE_NAMES
        }
        events = []

        # 架构级互斥。正常候选不会触发，这一层用于防御异常调用。
        if constrained["heater"] > 5.0 and constrained["vent"] > 5.0:
            if constrained["heater"] >= constrained["vent"]:
                constrained["vent"] = 0.0
            else:
                constrained["heater"] = 0.0
            events.append("thermal_mutual_exclusion")

        for name in DEVICE_NAMES:
            previous_value = float(previous.get(name, self._previous_command.get(name, 0.0)))
            desired_active = self._active(constrained[name])
            previous_active = self._active(previous_value)
            steps_since_switch = self._step - self._last_switch_step[name]
            if (
                name != "curtain"
                and desired_active != previous_active
                and steps_since_switch < self.minimum_hold_steps
            ):
                constrained[name] = previous_value
                events.append(f"{name}_minimum_hold")

            low = previous_value - self.MAX_DELTA[name]
            high = previous_value + self.MAX_DELTA[name]
            limited = _clip(constrained[name], max(0.0, low), min(100.0, high))
            if abs(limited - constrained[name]) > 1e-8:
                events.append(f"{name}_planned_rate_limit")
            constrained[name] = limited

        # 限速后再检查一次。冷热模式切换时，先把原模式平滑降到关闭，
        # 再允许新模式启动，避免为了互斥而瞬间越过变化率约束。
        if constrained["heater"] > 5.0 and constrained["vent"] > 5.0:
            if self._active(float(previous.get("heater", 0.0))):
                constrained["vent"] = 0.0
                events.append("vent_waits_for_heater_shutdown")
            elif self._active(float(previous.get("vent", 0.0))):
                constrained["heater"] = 0.0
                events.append("heater_waits_for_vent_shutdown")
            elif constrained["heater"] >= constrained["vent"]:
                constrained["vent"] = 0.0
                events.append("post_limit_heating_priority")
            else:
                constrained["heater"] = 0.0
                events.append("post_limit_ventilation_priority")

        return constrained, tuple(dict.fromkeys(events))

    def plan(
        self,
        action: Sequence[float],
        state: ControllerState,
        previous_command: Mapping[str, float],
    ) -> PlanResult:
        strategy = StrategyAction.from_normalized(action)
        previous = {
            name: _clip(float(previous_command.get(name, self._previous_command[name])), 0.0, 100.0)
            for name in DEVICE_NAMES
        }

        best = None
        for desired in self._candidate_commands(previous):
            command, events = self._apply_constraints(desired, previous)
            pred10, pred30, pred_rh30 = self._predict_candidate(
                state, command, previous
            )
            objective = self._objective(
                state,
                strategy,
                command,
                previous,
                pred10,
                pred30,
                pred_rh30,
            )
            candidate = (
                objective,
                sum(abs(command[name] - previous[name]) for name in DEVICE_NAMES),
                command,
                desired,
                pred10,
                pred30,
                pred_rh30,
                events,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if best is None:
            raise RuntimeError("V3滚动优化器未生成候选动作")

        objective, _, command, desired, pred10, pred30, pred_rh30, events = best
        for name in DEVICE_NAMES:
            was_active = self._active(previous[name])
            is_active = self._active(command[name])
            if was_active != is_active:
                self._last_switch_step[name] = self._step
        self._previous_command = dict(command)
        self._step += 1

        return PlanResult(
            strategy=strategy,
            command_pct=dict(command),
            unconstrained_command_pct=dict(desired),
            objective=float(objective),
            predicted_temperature_10min_c=float(pred10),
            predicted_temperature_30min_c=float(pred30),
            predicted_humidity_30min_pct=float(pred_rh30),
            constraint_events=events,
        )


def strategy_action(
    temperature_offset_c: float,
    comfort_weight: float,
    energy_weight: float,
    dehumidify_priority: float,
) -> np.ndarray:
    """供规则基线和测试脚本构造归一化策略动作。"""

    return StrategyAction(
        temperature_offset_c=_clip(float(temperature_offset_c), -2.0, 2.0),
        comfort_weight=_clip(float(comfort_weight), 0.0, 1.0),
        energy_weight=_clip(float(energy_weight), 0.0, 1.0),
        dehumidify_priority=_clip(float(dehumidify_priority), 0.0, 1.0),
    ).normalized()


if __name__ == "__main__":
    controller = StrategyToControlV3()
    controller.reset({name: 0.0 for name in DEVICE_NAMES})
    state = ControllerState(
        temperature_c=12.0,
        relative_humidity_pct=92.0,
        outdoor_temperature_c=2.0,
        outdoor_humidity_pct=65.0,
        solar_w_m2=0.0,
        predicted_10min_c=11.5,
        predicted_30min_c=10.0,
        target_low_c=8.0,
        target_high_c=10.0,
        is_day=False,
    )
    result = controller.plan(strategy_action(0.0, 0.8, 0.5, 0.8), state, {})
    print(result)
