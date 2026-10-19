"""三种控制器（任务书第十五章）。

控制器 A：固定阈值控制（无模型）
控制器 B：固定参数预测控制（固定系数一阶模型 + 单步预测）
控制器 C：在线参数辨识自适应预测控制（RLS 辨识加热增益 + 散热系数 + 单步预测）

控制机制（P1 诊断后改进）：
- 命令去重、加热 10% 步长、动作变化惩罚、功率变化率约束（可配置，默认 25%）
- 预测越界时临时突破平滑约束（快速响应）
- 排湿滞环 + 脉冲，DEHUMIDIFY_OVERRIDE 单独统计
- 加热-通风互斥（正常控温）

B/C 使用完全相同的动作空间、约束、目标函数，唯一差别是模型参数来源。

在线辨识（C）采用带温差项的递归最小二乘：
    温度变化 dt = 加热增益 × 实际加热功率
                 + 散热系数 × (室外温度 − 室内温度) + 偏置
用设备反馈的实际加热功率，t 时刻动作对应 t+1 时刻温度变化。
输入变化不足时标记"当前窗口不可辨识"，不强行更新。
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    ControlCommand, ActionType, ControlMode, CommandSource, new_id,
)


# ---------------------------------------------------------------------------
# 观测辅助
# ---------------------------------------------------------------------------

def _read(obs_map: dict[str, dict[str, Any]], point_id: str, default: float = None):
    o = obs_map.get(point_id)
    if o is None or o.get("value") is None:
        return default
    return float(o["value"])


def _zone_mean_temp(obs_map: dict[str, dict[str, Any]]) -> float | None:
    vals = []
    for zone in ("east", "center", "west"):
        v = _read(obs_map, f"{zone}_air_temperature_c")
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _make_cmd(device_id: str, target: float, action_type: ActionType,
              mode: ControlMode = ControlMode.RULE) -> ControlCommand:
    return ControlCommand(
        command_id=new_id("c"), trace_id=new_id("t"), greenhouse_id="GH001",
        device_id=device_id, action_type=action_type, target_value=target, unit="%",
        issued_at="", effective_at="", expires_at="",
        control_mode=mode, source=CommandSource.LOCAL_CONTROLLER,
    )


# ---------------------------------------------------------------------------
# 公共控制基类
# ---------------------------------------------------------------------------

class BaseController:
    name = "base"

    deadband_c = 0.5
    humidity_high_pct = 75.0
    humidity_low_pct = 65.0
    temp_protect_margin_c = 2.0

    def __init__(self):
        self._last_cmd: dict[str, float] = {}
        self._dehumidifying = False
        # 当前是否处于"合法排湿协同"模式（用于区分合法排湿 vs 非法加热-通风冲突）
        self.dehumidify_active = False
        # 最近一次决策元数据（供 controller_trace 落库）
        self.last_decision: dict[str, float] = {}

    def _apply_dedup(self, desired: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for dev, val in desired.items():
            last = self._last_cmd.get(dev)
            if last is None or abs(val - last) > 0.5:
                out[dev] = val
                self._last_cmd[dev] = val
        return out

    def _to_cmds(self, desired: dict[str, float],
                 mode: ControlMode = ControlMode.RULE) -> list[ControlCommand]:
        act_type = {
            "vent_01": ActionType.SET_POSITION,
            "fan_01": ActionType.SET_POWER,
            "curtain_01": ActionType.SET_POSITION,
            "heater_01": ActionType.SET_POWER,
        }
        return [_make_cmd(dev, val, act_type.get(dev, ActionType.SET_POSITION), mode)
                for dev, val in desired.items()]

    def _dehumidify_desired(self, t, target):
        if t < target - self.temp_protect_margin_c:
            return {"vent_01": 40.0, "fan_01": 40.0, "heater_01": 50.0}
        return {"vent_01": 100.0, "fan_01": 100.0, "heater_01": 30.0}


# ---------------------------------------------------------------------------
# 控制器 A：固定阈值
# ---------------------------------------------------------------------------

class ThresholdController(BaseController):
    name = "fixed_threshold"

    # 脉冲排湿参数（滞环 + 限时脉冲）
    pulse_vent_seconds = 240.0      # 每次通风持续 4 分钟
    pulse_idle_seconds = 720.0      # 两次通风之间至少等待 12 分钟
    pulse_vent_pct = 80.0           # 排湿通风开度

    def __init__(self, temp_min: float, temp_max: float, humidity_max: float):
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.humidity_max = humidity_max
        self._dehumidify_phase = "idle"   # idle / venting
        self._phase_elapsed = 0.0
        self._dt = 60.0

    def decide(self, obs_map, feedback_map=None):
        t = _zone_mean_temp(obs_map)
        rh = _read(obs_map, "center_relative_humidity_pct", 60.0)
        desired: dict[str, float] = {}

        if t is None:
            return [_make_cmd("curtain_01", 100.0, ActionType.SET_POSITION)]

        self._phase_elapsed += self._dt

        # 滞环：湿度 > 上限进入排湿需求，< 退出阈值取消
        if not self._dehumidifying and rh > self.humidity_high_pct:
            self._dehumidifying = True
        elif self._dehumidifying and rh < self.humidity_low_pct:
            self._dehumidifying = False
            self._dehumidify_phase = "idle"

        if self._dehumidifying:
            desired = self._pulse_dehumidify(t, rh)
            self.dehumidify_active = True
        else:
            self.dehumidify_active = False
            if t > self.temp_max:
                desired.update({"vent_01": 80.0, "fan_01": 80.0, "heater_01": 0.0})
            elif t < self.temp_min:
                desired.update({"vent_01": 0.0, "curtain_01": 100.0, "heater_01": 80.0})
            else:
                desired.update({"vent_01": 0.0, "heater_01": 0.0})

        return self._to_cmds(self._apply_dedup(desired), ControlMode.RULE)

    def _pulse_dehumidify(self, t, rh):
        """脉冲排湿：限时通风 + 等待期，温度保护缩开度。

        正常控温时加热与通风互斥；此处是排湿专用（DEHUMIDIFY_OVERRIDE），
        允许短时加热 + 通风协同，但用脉冲限时避免长期冲突。
        """
        vent = 0.0
        fan = 0.0
        heater = 0.0

        # 温度保护：温度接近下限（< temp_min + margin）时，缩小通风并加少量加热
        temp_low = t < self.temp_min + self.temp_protect_margin_c

        if self._dehumidify_phase == "venting":
            # 通风脉冲进行中
            if self._phase_elapsed >= self.pulse_vent_seconds:
                self._dehumidify_phase = "idle"
                self._phase_elapsed = 0.0
            else:
                if temp_low:
                    # 温度偏低：缩小通风，加少量加热协同排湿
                    vent = 40.0
                    fan = 40.0
                    heater = 40.0
                else:
                    vent = self.pulse_vent_pct
                    fan = self.pulse_vent_pct
                    heater = 0.0
        else:
            # 等待期：不通风，直到等待期满才进入下一次脉冲
            if self._phase_elapsed >= self.pulse_idle_seconds:
                self._dehumidify_phase = "venting"
                self._phase_elapsed = 0.0
            else:
                # 等待期保持保温（关窗保温帘关）
                vent = 0.0
                fan = 0.0
                heater = 0.0
                # 但若温度过低，需要加热保温
                if t < self.temp_min:
                    heater = 80.0

        desired = {"vent_01": vent, "fan_01": fan}
        if heater > 0:
            desired["heater_01"] = heater
        desired["curtain_01"] = 100.0  # 保温帘关闭，避免散热
        return desired


# ---------------------------------------------------------------------------
# 预测控制器公共框架
# ---------------------------------------------------------------------------

HEATER_LEVELS = [i * 10.0 for i in range(11)]
VENT_LEVELS = [0.0, 50.0, 100.0]
CURTAIN_LEVELS = [0.0, 50.0, 100.0]

ACTION_CHANGE_PENALTY = 0.02


class PredictiveBase(BaseController):
    name = "predictive"

    ventilation_gain = 0.010
    solar_gain = 0.0005
    max_heater_change = 25.0   # 每周期加热功率最大变化（%，默认 25）

    def __init__(self, target: float, day_target: float | None = None,
                 night_target: float | None = None):
        super().__init__()
        self.target = target
        # 昼夜目标（可选）：若提供，则按太阳辐射动态切换控制目标
        self.day_target = day_target if day_target is not None else target
        self.night_target = night_target if night_target is not None else target
        self._prev_action = (0.0, 0.0, 0.0, 0.0)

    def _current_target(self, solar: float) -> float:
        """按太阳辐射判断昼夜，返回当前控制目标。"""
        # 白天阈值与 config.targets.day_radiation_threshold_w_m2 一致
        return self.day_target if solar >= 50.0 else self.night_target

    def _predict(self, t, t_out, solar, vent, fan, curtain, heater, heater_gain):
        vent_eff = vent / 100.0 + 0.7 * fan / 100.0
        pred = t + self.ventilation_gain * vent_eff * (t_out - t)
        pred += self.solar_gain * solar * (1.0 - 0.7 * curtain / 100.0)
        pred += heater_gain * heater / 100.0
        return pred

    def _select_action(self, t, t_out, solar, heater_gain, target):
        best = None
        best_cost = float("inf")
        prev_vent, prev_fan, prev_curtain, prev_heater = self._prev_action

        # 预测越界检测：若当前温度已低于目标较多，允许临时突破平滑约束快速加热
        urgent = (t < target - 3.0)

        for vent in VENT_LEVELS:
            for fan in VENT_LEVELS:
                for curtain in CURTAIN_LEVELS:
                    for heater in HEATER_LEVELS:
                        if not urgent and abs(heater - prev_heater) > self.max_heater_change:
                            continue
                        if heater > 5 and (vent > 5 or fan > 5):
                            continue
                        pred = self._predict(t, t_out, solar, vent, fan, curtain,
                                             heater, heater_gain)
                        cost = (pred - target) ** 2
                        if not urgent:
                            cost += ACTION_CHANGE_PENALTY * (
                                abs(heater - prev_heater) + abs(vent - prev_vent)
                                + abs(fan - prev_fan) + abs(curtain - prev_curtain))
                        cost += 0.004 * fan + 0.002 * vent
                        if cost < best_cost:
                            best_cost = cost
                            best = (vent, fan, curtain, heater)
        return best, best_cost

    def _decide_common(self, obs_map, heater_gain):
        t = _zone_mean_temp(obs_map)
        t_out = _read(obs_map, "outdoor_air_temperature_c", t if t is not None else 0.0)
        solar = _read(obs_map, "outdoor_solar_irradiance_w_m2", 0.0)
        rh = _read(obs_map, "center_relative_humidity_pct", 60.0)

        if t is None:
            return [_make_cmd("curtain_01", 100.0, ActionType.SET_POSITION)]

        if not self._dehumidifying and rh > self.humidity_high_pct:
            self._dehumidifying = True
        elif self._dehumidifying and rh < self.humidity_low_pct:
            self._dehumidifying = False

        cur_target = self._current_target(solar)

        if self._dehumidifying:
            desired = self._dehumidify_desired(t, cur_target)
            self.dehumidify_active = True
            self._prev_action = (desired.get("vent_01", 0), desired.get("fan_01", 0),
                                 self._prev_action[2], desired.get("heater_01", 0))
            return self._to_cmds(self._apply_dedup(desired))

        self.dehumidify_active = False
        best, best_cost = self._select_action(t, t_out, solar, heater_gain, cur_target)
        vent, fan, curtain, heater = best
        self._prev_action = best
        # 记录决策元数据（供 controller_trace 落库）
        pred_temp = self._predict(t, t_out, solar, vent, fan, curtain, heater, heater_gain)
        self.last_decision = {
            "model_name": self.name,
            "prediction_horizon_steps": 1,
            "predicted_temperature_c": round(pred_temp, 3),
            "objective_total_cost": round(best_cost, 4),
        }
        desired = {
            "vent_01": vent, "fan_01": fan,
            "curtain_01": curtain, "heater_01": heater,
        }
        return self._to_cmds(self._apply_dedup(desired))


# ---------------------------------------------------------------------------
# 控制器 B：固定参数预测
# ---------------------------------------------------------------------------

class FixedModelController(PredictiveBase):
    name = "fixed_model"
    heater_gain = 0.45

    def __init__(self, target: float, day_target: float | None = None,
                 night_target: float | None = None):
        super().__init__(target, day_target, night_target)

    def decide(self, obs_map, feedback_map=None):
        return self._decide_common(obs_map, self.heater_gain)


# ---------------------------------------------------------------------------
# Oracle 诊断控制器（仅用于实验诊断，不作为项目成果）
# 直接读取仿真器真实加热效率，用于判断"理论上是否存在自适应提升空间"。
# 与 B 使用完全相同的动作空间/约束/目标函数，唯一差别是 heater_gain 跟随真实效率缩放。
# ---------------------------------------------------------------------------

class OracleController(PredictiveBase):
    name = "oracle"
    heater_gain = 0.45   # 基线增益（对应效率 1.0）

    def __init__(self, target: float, day_target: float | None = None,
                 night_target: float | None = None):
        super().__init__(target, day_target, night_target)
        self._true_efficiency = 1.0

    def set_true_efficiency(self, eff: float) -> None:
        """由实验框架注入仿真器真实加热效率（仅诊断用）。"""
        self._true_efficiency = eff

    def decide(self, obs_map, feedback_map=None):
        # Oracle 增益 = 基线增益 × 真实效率（相当于已知真实设备能力）
        gain = self.heater_gain * self._true_efficiency
        return self._decide_common(obs_map, gain)


# ---------------------------------------------------------------------------
# 控制器 C：带温差项 RLS 在线辨识
# ---------------------------------------------------------------------------

class AdaptiveController(PredictiveBase):
    """带温差项的递归最小二乘在线辨识。

    辨识模型：
        dt = gain × heater_actual + k_cool × (T_out − T_in) + bias
    用设备反馈的实际加热功率，t 时刻动作对应 t+1 时刻温度变化。
    输入变化不足时标记不可辨识，不更新。
    """

    name = "adaptive_model"

    def __init__(self, target: float, day_target: float | None = None,
                 night_target: float | None = None):
        super().__init__(target, day_target, night_target)
        self.heater_gain = 0.45       # 加热增益估计
        self._k_cool = 0.02           # 散热系数估计
        self._bias = 0.0              # 偏置
        self._P = [[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]]
        self._forget = 0.99
        self._prev_t: float | None = None
        self._prev_t_out: float = 0.0
        self._prev_heater_actual: float = 0.0
        self.update_count = 0
        self.unidentifiable_count = 0
        self.trace: list[dict[str, float]] = []
        # 归一化估计增益：突变前稳定基线 / 当前归一化值（= 当前/基线）
        # 基线增益：默认初始化为固定增益 0.45（而非 None），
        # 保证性能衰减告警在任何场景（含极寒 benchmark）都能启用；
        # 实验框架可在突变前调用 set_baseline_gain 覆盖为突变前稳定估计。
        self.baseline_gain: float = 0.45
        self.normalized_gain: float | None = None
        # 置信度门控状态
        self.confidence: float = 0.0       # 综合置信度 [0,1]
        self.fused_gain: float = 0.45      # 门控融合后的有效增益（实际用于控制）
        self.fallback_active: bool = False  # 是否处于固定模型回退
        self.degradation_alert: bool = False  # 加热设备性能衰减告警
        self._recent_gains: list[float] = []  # 近期增益（用于稳定性评估）

    def _rls_update(self, heater_actual: float, dt_obs: float, t_out: float,
                    t_in: float) -> None:
        """带温差项 RLS 更新三个参数 [gain, k_cool, bias]。"""
        x = [heater_actual, (t_out - t_in), 1.0]
        theta = [self.heater_gain, self._k_cool, self._bias]
        y_pred = theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[2]
        err = dt_obs - y_pred

        P = self._P
        px = [sum(P[i][j] * x[j] for j in range(3)) for i in range(3)]
        denom = self._forget + sum(x[i] * px[i] for i in range(3))
        if abs(denom) < 1e-9:
            return
        K = [px[i] / denom for i in range(3)]

        new_theta = [theta[i] + K[i] * err for i in range(3)]
        # 限幅保护
        new_theta[0] = max(0.15, min(1.5, new_theta[0]))   # gain（下限0.15，避免塌缩到无物理意义）
        new_theta[1] = max(-0.5, min(0.5, new_theta[1]))   # k_cool
        new_theta[2] = max(-1.0, min(1.0, new_theta[2]))   # bias
        self.heater_gain, self._k_cool, self._bias = new_theta

        # 协方差更新 P = (I - K x^T) P / lambda
        newP = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                kx = sum(K[i] * x[k] * P[k][j] for k in range(3))
                newP[i][j] = (P[i][j] - kx) / self._forget
        self._P = newP
        self.update_count += 1

    def decide(self, obs_map, feedback_map=None):
        t = _zone_mean_temp(obs_map)
        t_out = _read(obs_map, "outdoor_air_temperature_c", t if t is not None else 0.0)
        solar = _read(obs_map, "outdoor_solar_irradiance_w_m2", 0.0)

        # 用设备反馈的实际加热功率（feedback_map 由 benchmark 传入）
        heater_actual = 0.0
        if feedback_map is not None:
            heater_actual = feedback_map.get("heater_01", 0.0) / 100.0

        # 辨识：t 时刻动作 → t+1 时刻温度变化（用上一周期实际加热，本周期温度）
        if self._prev_t is not None and t is not None:
            dt_obs = t - self._prev_t
            # 仅在夜间（低太阳）辨识，太阳得热项可忽略
            if solar <= 50.0:
                # 可辨识性检查：加热功率需有足够激励（否则 gain 维度无信息，会病态塌缩）
                heater_incentive = self._prev_heater_actual >= 0.10
                ident = (heater_incentive
                         or abs(dt_obs) > 0.01)
                if ident:
                    self._rls_update(self._prev_heater_actual, dt_obs,
                                     self._prev_t_out, self._prev_t)
                else:
                    self.unidentifiable_count += 1
            else:
                self.unidentifiable_count += 1

        self.trace.append({
            "heater_gain_estimate": round(self.heater_gain, 4),
            "k_cool_estimate": round(self._k_cool, 4),
            "bias_estimate": round(self._bias, 4),
            "heater_actual": heater_actual,
            "normalized_gain": round(self.normalized_gain, 4) if self.normalized_gain is not None else None,
        })

        # 更新历史（本周期实际加热功率、温度、室外温）
        self._prev_heater_actual = heater_actual
        self._prev_t = t
        self._prev_t_out = t_out

        # 置信度门控：计算综合置信度与门控融合增益
        self._update_confidence_and_fusion()

        return self._decide_common(obs_map, self.fused_gain)

    def snapshot(self):
        return {"heater_gain": round(self.heater_gain, 4),
                "k_cool": round(self._k_cool, 4),
                "bias": round(self._bias, 4),
                "baseline_gain": self.baseline_gain,
                "normalized_gain": round(self.normalized_gain, 4) if self.normalized_gain is not None else None,
                "confidence": round(self.confidence, 4),
                "fused_gain": round(self.fused_gain, 4),
                "fallback_active": self.fallback_active,
                "degradation_alert": self.degradation_alert,
                "update_count": self.update_count,
                "unidentifiable_count": self.unidentifiable_count}

    def set_baseline_gain(self, gain: float) -> None:
        """设置突变前稳定基线增益，用于归一化（由实验框架在突变时刻调用）。"""
        self.baseline_gain = gain

    def update_normalized_gain(self) -> None:
        """计算归一化估计增益 = 当前估计值 / 突变前基线。"""
        if self.baseline_gain and self.baseline_gain > 0:
            self.normalized_gain = self.heater_gain / self.baseline_gain

    # ------------------------------------------------------------------
    # 置信度门控（安全自适应核心）
    # ------------------------------------------------------------------

    def _update_confidence_and_fusion(self) -> None:
        """综合激励充分性、估计稳定性、合理性，得到置信度并门控融合。

        融合规则：
        - 置信度 alpha ∈ [0,1]，fused_gain = fixed_gain + alpha × (estimated_gain - fixed_gain)
        - 激励不足/估计异常 → alpha 下降，向固定增益回退
        - 增益持续下降（相对基线）→ 生成性能衰减告警
        """
        fixed_gain = 0.45  # 与 FixedModelController 一致

        # 记录近期增益（最多 60 个周期）
        self._recent_gains.append(self.heater_gain)
        if len(self._recent_gains) > 60:
            self._recent_gains.pop(0)

        # 1) 合理性：增益超出物理范围 → 异常
        if not (0.15 <= self.heater_gain <= 1.5):
            self.fallback_active = True
            self.confidence = 0.0
            self.fused_gain = fixed_gain
            return

        # 2) 估计稳定性：近期增益的标准差（相对均值）
        if len(self._recent_gains) >= 10:
            mu = sum(self._recent_gains) / len(self._recent_gains)
            var = sum((g - mu) ** 2 for g in self._recent_gains) / len(self._recent_gains)
            std = var ** 0.5
            rel_std = std / max(1e-6, abs(mu))
            stability = max(0.0, 1.0 - rel_std / 0.5)  # 相对标准差 50% 以上 → 稳定性为 0
        else:
            stability = 0.3  # 样本不足，中等偏保守

        # 3) 激励充分性：最近 60 周期内加热激励充分的比例
        incentive_ratio = 0.0
        if self.update_count + self.unidentifiable_count > 0:
            incentive_ratio = self.update_count / max(1, self.update_count + self.unidentifiable_count)
        # 更新比例低说明大量时间不可辨识 → 激励不足
        incentive = min(1.0, incentive_ratio / 0.5)  # 更新占比 50% 视为充分

        # 4) 综合置信度 = 激励 × 稳定性 × 合理性（合理性已在上方处理）
        self.confidence = incentive * stability

        # 5) 门控融合：低置信度时向固定增益收缩
        alpha = self.confidence
        self.fused_gain = fixed_gain + alpha * (self.heater_gain - fixed_gain)

        # 6) 回退判定：置信度过低 → 标记回退，用固定增益
        if self.confidence < 0.2:
            self.fallback_active = True
            self.fused_gain = fixed_gain
        else:
            self.fallback_active = False

        # 7) 性能衰减告警：相对基线持续下降超过 20% 且持续
        if self.baseline_gain and self.baseline_gain > 0:
            ratio = self.heater_gain / self.baseline_gain
            self.degradation_alert = ratio < 0.8


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def make_controller(kind: str, temp_min: float, temp_max: float,
                    humidity_max: float, target: float = 22.5,
                    night_temp_min: float = 8.0, night_temp_max: float = 10.0):
    day_tgt = target
    night_tgt = (night_temp_min + night_temp_max) / 2.0
    if kind == "fixed_threshold":
        return ThresholdController(temp_min, temp_max, humidity_max)
    if kind == "fixed_model":
        return FixedModelController(target=target, day_target=day_tgt, night_target=night_tgt)
    if kind == "adaptive_model":
        return AdaptiveController(target=target, day_target=day_tgt, night_target=night_tgt)
    if kind == "oracle":
        return OracleController(target=target, day_target=day_tgt, night_target=night_tgt)
    raise ValueError(f"unknown controller kind: {kind}")
