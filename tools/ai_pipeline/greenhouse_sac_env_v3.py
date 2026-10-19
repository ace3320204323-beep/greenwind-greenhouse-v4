"""绿风润耕 SAC V3 分层控制环境。

V3 保留 V2 的虚拟大棚、Transformer 状态和安全保护，但改变 SAC 动作语义：
SAC 只输出温度目标修正、舒适权重、节能权重和排湿优先级；具体设备动作
由 StrategyToControlV3 通过约束滚动优化生成。

该环境依赖同目录下已经跑通的 greenhouse_sac_env.py、
strategy_to_control_v3.py、Transformer 模型和虚拟大棚模块。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from greenhouse_sac_env import GreenhouseSACEnv
from strategy_to_control_v3 import (
    ControllerState,
    PlanResult,
    STRATEGY_NAMES,
    StrategyAction,
    StrategyToControlV3,
)


class GreenhouseSACV3Env(GreenhouseSACEnv):
    """SAC 上层策略与下层约束控制器组成的单棚训练环境。"""

    STRATEGY_NAMES = STRATEGY_NAMES
    OBSERVATION_NAMES = GreenhouseSACEnv.OBSERVATION_NAMES + (
        "previous_temperature_offset_norm",
        "previous_comfort_weight_norm",
        "previous_energy_weight_norm",
        "previous_dehumidify_priority_norm",
    )

    def __init__(self, *args: Any, minimum_hold_steps: int = 2, **kwargs: Any) -> None:
        self._strategy_controller = StrategyToControlV3(
            minimum_hold_steps=minimum_hold_steps
        )
        self._previous_strategy = StrategyAction.from_normalized(
            np.zeros(4, dtype=np.float32)
        )
        self._strategy_movement_norm = 0.0
        self._latest_plan: Optional[PlanResult] = None
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        self._previous_strategy = StrategyAction.from_normalized(
            np.zeros(4, dtype=np.float32)
        )
        self._strategy_movement_norm = 0.0
        self._latest_plan = None
        _, info = super().reset(**kwargs)
        self._strategy_controller.reset(self._previous_targets)
        # parent.reset 在热身结束后已更新设备状态；此处重新生成带策略历史的观测。
        observation = self._make_observation()
        info = self._make_info(
            requested_targets=self._previous_targets,
            safe_targets=self._previous_targets,
            safety_overrides=(),
            reward_components={},
        )
        self._last_info = info
        return observation, info

    def _controller_state(self) -> ControllerState:
        record = self._latest_record
        center = float(record.get("t_center_c", 15.0))
        low, high = self._target_band()
        return ControllerState(
            temperature_c=center,
            relative_humidity_pct=float(record.get("rh_center_pct", 70.0)),
            outdoor_temperature_c=float(record.get("out_temp_c", 0.0)),
            outdoor_humidity_pct=float(record.get("out_rh_pct", 60.0)),
            solar_w_m2=float(record.get("out_solar_w_m2", 0.0)),
            predicted_10min_c=self._prediction_value("10min", center),
            predicted_30min_c=self._prediction_value("30min", center),
            target_low_c=float(low),
            target_high_c=float(high),
            is_day=bool(record.get("is_day", 0)),
        )

    def _decode_action(self, action: np.ndarray) -> Dict[str, float]:
        strategy = StrategyAction.from_normalized(action)
        previous_normalized = self._previous_strategy.normalized()
        current_normalized = strategy.normalized()
        self._strategy_movement_norm = float(
            np.mean(np.abs(current_normalized - previous_normalized)) / 2.0
        )
        self._latest_plan = self._strategy_controller.plan(
            action=action,
            state=self._controller_state(),
            previous_command=self._previous_targets,
        )
        self._previous_strategy = strategy
        return dict(self._latest_plan.command_pct)

    def _calculate_reward(
        self,
        safe_movement: float,
        raw_request_movement: float,
        energy_norm: float,
        unsafe_request: float,
        conflict_intensity: float,
    ) -> Tuple[float, Dict[str, float]]:
        reward, components = super()._calculate_reward(
            safe_movement=safe_movement,
            raw_request_movement=raw_request_movement,
            energy_norm=energy_norm,
            unsafe_request=unsafe_request,
            conflict_intensity=conflict_intensity,
        )
        strategy_penalty = -0.10 * self._strategy_movement_norm
        planner_events = (
            len(self._latest_plan.constraint_events)
            if self._latest_plan is not None
            else 0
        )
        # 下层限速属于计划内约束，不作为安全故障；仅轻微抑制频繁触碰约束。
        planner_penalty = -0.01 * min(planner_events, 4)
        reward = float(np.clip(reward + strategy_penalty + planner_penalty, -20.0, 1.0))
        components["strategy_movement_penalty"] = strategy_penalty
        components["planner_constraint_penalty"] = planner_penalty
        components["strategy_movement_norm"] = self._strategy_movement_norm
        components["planner_constraint_event_count"] = float(planner_events)
        components["reward_version"] = 3.0
        return reward, components

    def _make_observation(self) -> np.ndarray:
        base = super()._make_observation()
        strategy_history = self._previous_strategy.normalized().astype(np.float32)
        vector = np.concatenate([base, strategy_history]).astype(np.float32)
        return np.clip(np.nan_to_num(vector), -10.0, 10.0).astype(np.float32)

    def _make_info(
        self,
        requested_targets: Mapping[str, float],
        safe_targets: Mapping[str, float],
        safety_overrides: Iterable[str],
        reward_components: Mapping[str, float],
    ) -> Dict[str, Any]:
        info = super()._make_info(
            requested_targets=requested_targets,
            safe_targets=safe_targets,
            safety_overrides=safety_overrides,
            reward_components=reward_components,
        )
        plan = self._latest_plan
        info.update(
            {
                "architecture": "transformer_sac_strategy_constrained_control_v3",
                "strategy_action": self._previous_strategy.as_dict(),
                "strategy_movement_norm": self._strategy_movement_norm,
                "planner_constraint_events": (
                    plan.constraint_events if plan is not None else ()
                ),
                "planner_objective": (
                    float(plan.objective) if plan is not None else 0.0
                ),
                "planner_unconstrained_command_pct": (
                    dict(plan.unconstrained_command_pct)
                    if plan is not None
                    else dict(requested_targets)
                ),
                "planner_pred_10min_c": (
                    float(plan.predicted_temperature_10min_c)
                    if plan is not None
                    else float(info["pred_10min_c"])
                ),
                "planner_pred_30min_c": (
                    float(plan.predicted_temperature_30min_c)
                    if plan is not None
                    else float(info["pred_30min_c"])
                ),
                "planner_pred_rh_30min_pct": (
                    float(plan.predicted_humidity_30min_pct)
                    if plan is not None
                    else float(info["relative_humidity_pct"])
                ),
            }
        )
        return info


if __name__ == "__main__":
    env = GreenhouseSACV3Env(episode_hours=2.0, render_mode="ansi")
    try:
        observation, info = env.reset(seed=42)
        print("observation_shape=", observation.shape)
        print("action_shape=", env.action_space.shape)
        conflicts = 0
        for _ in range(20):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            safe = info["safe_targets_pct"]
            conflicts += int(safe["heater"] > 5.0 and safe["vent"] > 5.0)
            if terminated or truncated:
                break
        print("random_steps=", info["decision_step"])
        print("heater_vent_conflicts=", conflicts)
        print("last_strategy=", info["strategy_action"])
    finally:
        env.close()
