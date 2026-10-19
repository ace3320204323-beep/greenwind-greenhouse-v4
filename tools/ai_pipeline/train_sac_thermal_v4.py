"""在热惯性虚拟大棚上训练绿风润耕 SAC V4 策略。

本脚本只训练上层 SAC 策略。SAC 输出四个策略量：温度目标修正、舒适权重、
节能权重和排湿优先级；下层 StrategyToControlV3 再把策略转换为加热、通风、
卷帘和风机的设备指令，并经过安全约束后送入虚拟大棚。

默认加载 outputs_thermal_v3 中重新训练的 Transformer，因此不会误用加入热惯性
以前的预测模型。旧的 formal_v3 权重不会被覆盖。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


AI_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_PIPELINE_DIR.parent.parent
DEFAULT_SCENARIO = (
    PROJECT_ROOT
    / "tools"
    / "virtual_greenhouse"
    / "scenarios"
    / "shandong_winter_strawberry.json"
)
DEFAULT_TRANSFORMER_MODEL = (
    AI_PIPELINE_DIR / "outputs_thermal_v3" / "transformer_v2.pt"
)
DEFAULT_TRANSFORMER_META = (
    AI_PIPELINE_DIR / "outputs_thermal_v3" / "transformer_v2_meta.json"
)
DEFAULT_OUTPUT = AI_PIPELINE_DIR / "sac_runs" / "thermal_v4"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _distance_to_band(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


@dataclass
class EpisodeRow:
    episode: int
    total_timesteps: int
    episode_steps: int
    reward: float
    mean_temperature_error_c: float
    temperature_in_band_rate: float
    mean_humidity_error_pct: float
    safety_override_rate: float
    planner_constraint_rate: float
    mean_load_power_kw: float
    mean_grid_power_kw: float
    mean_battery_soc_pct: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在热惯性虚拟大棚中训练 SAC V4 上层策略"
    )
    parser.add_argument("--total-timesteps", type=int, default=80_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-hours", type=float, default=72.0)
    parser.add_argument("--control-interval-minutes", type=int, default=5)
    parser.add_argument("--warmup-minutes", type=int, default=60)
    parser.add_argument("--minimum-hold-steps", type=int, default=2)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--transformer-model", type=Path, default=DEFAULT_TRANSFORMER_MODEL
    )
    parser.add_argument(
        "--transformer-meta", type=Path, default=DEFAULT_TRANSFORMER_META
    )
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--final-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cpu",
        help="默认 cpu；如已安装 CUDA 版 PyTorch 可手动传 cuda",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="消融实验使用：关闭 Transformer，预测值退化为当前温度",
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps 必须大于 0")
    if args.episode_hours <= 0:
        raise ValueError("--episode-hours 必须大于 0")
    if args.control_interval_minutes < 1:
        raise ValueError("--control-interval-minutes 必须大于等于 1")
    if not args.scenario.exists():
        raise FileNotFoundError(f"未找到虚拟大棚场景：{args.scenario}")
    if not args.no_transformer:
        if not args.transformer_model.exists():
            raise FileNotFoundError(
                f"未找到热惯性 Transformer 权重：{args.transformer_model}"
            )
        if not args.transformer_meta.exists():
            raise FileNotFoundError(
                f"未找到热惯性 Transformer 元数据：{args.transformer_meta}"
            )


def load_dependencies() -> Mapping[str, Any]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "缺少 SAC 训练依赖。请在 greenwind_ai 环境中执行：\n"
            "python -m pip install stable-baselines3 gymnasium matplotlib"
        ) from exc

    from greenhouse_sac_env_v3 import GreenhouseSACV3Env

    return {
        "SAC": SAC,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "GreenhouseSACV3Env": GreenhouseSACV3Env,
    }


def build_episode_callback(base_callback: Any, csv_path: Path) -> Any:
    """动态构造 SB3 回调，避免未安装 SB3 时脚本连 --help 都无法使用。"""

    class EpisodeMetricsCallback(base_callback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.csv_path = csv_path
            self.rows: List[EpisodeRow] = []
            self._episode_number = 0
            self._reset_accumulator()

        def _reset_accumulator(self) -> None:
            self._rewards: List[float] = []
            self._temperature_errors: List[float] = []
            self._temperature_in_band: List[float] = []
            self._humidity_errors: List[float] = []
            self._overrides: List[float] = []
            self._planner_constraints: List[float] = []
            self._load_power: List[float] = []
            self._grid_power: List[float] = []
            self._battery_soc: List[float] = []

        @staticmethod
        def _mean(values: Sequence[float]) -> float:
            return float(mean(values)) if values else 0.0

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
            dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
            infos = self.locals.get("infos", [{}])

            # 当前训练固定为单环境。保留数组读取方式，兼容 DummyVecEnv。
            reward = float(rewards[0]) if len(rewards) else 0.0
            done = bool(dones[0]) if len(dones) else False
            info = infos[0] if infos else {}
            low, high = info.get("target_band_c", (0.0, 0.0))
            temperature = float(info.get("temperature_c", 0.0))
            temperature_error = _distance_to_band(
                temperature, float(low), float(high)
            )
            rh = float(info.get("relative_humidity_pct", 0.0))
            humidity_error = _distance_to_band(rh, 60.0, 85.0)
            energy = info.get("energy", {}) or {}

            self._rewards.append(reward)
            self._temperature_errors.append(temperature_error)
            self._temperature_in_band.append(float(temperature_error == 0.0))
            self._humidity_errors.append(humidity_error)
            self._overrides.append(float(bool(info.get("safety_overrides", ()))))
            self._planner_constraints.append(
                float(bool(info.get("planner_constraint_events", ())))
            )
            self._load_power.append(float(energy.get("load_power_kw", 0.0)))
            self._grid_power.append(float(energy.get("grid_power_kw", 0.0)))
            self._battery_soc.append(float(energy.get("battery_soc_pct", 0.0)))

            if done:
                self._episode_number += 1
                row = EpisodeRow(
                    episode=self._episode_number,
                    total_timesteps=int(self.num_timesteps),
                    episode_steps=len(self._rewards),
                    reward=float(sum(self._rewards)),
                    mean_temperature_error_c=self._mean(self._temperature_errors),
                    temperature_in_band_rate=self._mean(self._temperature_in_band),
                    mean_humidity_error_pct=self._mean(self._humidity_errors),
                    safety_override_rate=self._mean(self._overrides),
                    planner_constraint_rate=self._mean(self._planner_constraints),
                    mean_load_power_kw=self._mean(self._load_power),
                    mean_grid_power_kw=self._mean(self._grid_power),
                    mean_battery_soc_pct=self._mean(self._battery_soc),
                )
                self.rows.append(row)
                self.save()
                print(
                    f"episode={row.episode} reward={row.reward:.2f} "
                    f"temp_error={row.mean_temperature_error_c:.3f}C "
                    f"in_band={row.temperature_in_band_rate:.1%} "
                    f"override={row.safety_override_rate:.1%}"
                )
                self._reset_accumulator()
            return True

        def save(self) -> None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.rows:
                return
            fieldnames = list(asdict(self.rows[0]).keys())
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.rows:
                    writer.writerow(asdict(row))

    return EpisodeMetricsCallback()


def plot_training(rows: Sequence[EpisodeRow], output_path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过训练曲线。")
        return

    episodes = np.asarray([row.episode for row in rows])
    rewards = np.asarray([row.reward for row in rows], dtype=float)
    errors = np.asarray([row.mean_temperature_error_c for row in rows], dtype=float)
    in_band = np.asarray([row.temperature_in_band_rate for row in rows], dtype=float)
    overrides = np.asarray([row.safety_override_rate for row in rows], dtype=float)

    def rolling(values: np.ndarray, window: int = 10) -> np.ndarray:
        if len(values) < 2:
            return values.copy()
        result = []
        for index in range(len(values)):
            start = max(0, index - window + 1)
            result.append(float(np.mean(values[start : index + 1])))
        return np.asarray(result)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SAC V4 thermal-inertia training", fontsize=15)

    axes[0, 0].plot(episodes, rewards, alpha=0.28, label="episode")
    axes[0, 0].plot(episodes, rolling(rewards), linewidth=2, label="rolling mean")
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, errors, alpha=0.32)
    axes[0, 1].plot(episodes, rolling(errors), linewidth=2)
    axes[0, 1].set_title("Temperature error to target band (C)")

    axes[1, 0].plot(episodes, in_band * 100.0, alpha=0.32)
    axes[1, 0].plot(episodes, rolling(in_band) * 100.0, linewidth=2)
    axes[1, 0].set_title("Temperature in-band rate (%)")

    axes[1, 1].plot(episodes, overrides * 100.0, alpha=0.32)
    axes[1, 1].plot(episodes, rolling(overrides) * 100.0, linewidth=2)
    axes[1, 1].set_title("Safety override rate (%)")

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
        axis.set_xlabel("Episode")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model: Any,
    env_class: Any,
    args: argparse.Namespace,
    episodes: int,
) -> Dict[str, Any]:
    episode_rows: List[Dict[str, float]] = []
    for episode_index in range(episodes):
        env = env_class(
            scenario_path=args.scenario,
            model_path=args.transformer_model,
            meta_path=args.transformer_meta,
            episode_hours=args.episode_hours,
            control_interval_minutes=args.control_interval_minutes,
            warmup_minutes=args.warmup_minutes,
            use_transformer=not args.no_transformer,
            minimum_hold_steps=args.minimum_hold_steps,
        )
        try:
            observation, _ = env.reset(seed=args.seed + 10_000 + episode_index)
            rewards: List[float] = []
            temp_errors: List[float] = []
            in_band: List[float] = []
            overrides: List[float] = []
            planner_constraints: List[float] = []
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                low, high = info["target_band_c"]
                error = _distance_to_band(
                    float(info["temperature_c"]), float(low), float(high)
                )
                rewards.append(float(reward))
                temp_errors.append(error)
                in_band.append(float(error == 0.0))
                overrides.append(float(bool(info.get("safety_overrides", ()))))
                planner_constraints.append(
                    float(bool(info.get("planner_constraint_events", ())))
                )
                if terminated or truncated:
                    break
            episode_rows.append(
                {
                    "seed": args.seed + 10_000 + episode_index,
                    "reward": float(sum(rewards)),
                    "mean_temperature_error_c": float(mean(temp_errors)),
                    "temperature_in_band_rate": float(mean(in_band)),
                    "safety_override_rate": float(mean(overrides)),
                    "planner_constraint_rate": float(mean(planner_constraints)),
                }
            )
        finally:
            env.close()

    keys = [key for key in episode_rows[0].keys() if key != "seed"]
    aggregate = {
        key: {
            "mean": float(mean(float(row[key]) for row in episode_rows)),
            "std": float(pstdev(float(row[key]) for row in episode_rows)),
        }
        for key in keys
    }
    return {
        "evaluation_episodes": episodes,
        "deterministic_policy": True,
        "episodes": episode_rows,
        "aggregate": aggregate,
        "metric_note": (
            "mean_temperature_error_c 是棚温到昼夜目标区间的距离，"
            "不是 Transformer 预测误差。"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out = args.out.resolve()
    args.scenario = args.scenario.resolve()
    args.transformer_model = args.transformer_model.resolve()
    args.transformer_meta = args.transformer_meta.resolve()
    validate_inputs(args)
    deps = load_dependencies()
    args.out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "architecture": "thermal_inertia_transformer_sac_v3_constrained_control",
        "algorithm": "SAC",
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "episode_hours": args.episode_hours,
        "control_interval_minutes": args.control_interval_minutes,
        "warmup_minutes": args.warmup_minutes,
        "minimum_hold_steps": args.minimum_hold_steps,
        "use_transformer": not args.no_transformer,
        "scenario": str(args.scenario),
        "transformer_model": str(args.transformer_model),
        "transformer_meta": str(args.transformer_meta),
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "device": args.device,
        "action_semantics": [
            "temperature_offset_c",
            "comfort_weight",
            "energy_weight",
            "dehumidify_priority",
        ],
        "boundary": (
            "本次训练在热惯性虚拟大棚中完成；结果用于仿真验证和系统联调，"
            "不等同于真实棚室效果。"
        ),
    }
    _json_dump(args.out / "run_config.json", run_config)
    print("开始训练：")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    env_class = deps["GreenhouseSACV3Env"]
    monitor_class = deps["Monitor"]
    dummy_vec_env = deps["DummyVecEnv"]

    def make_env(seed_offset: int, monitor_file: Optional[Path] = None) -> Any:
        def factory() -> Any:
            env = env_class(
                scenario_path=args.scenario,
                model_path=args.transformer_model,
                meta_path=args.transformer_meta,
                episode_hours=args.episode_hours,
                control_interval_minutes=args.control_interval_minutes,
                warmup_minutes=args.warmup_minutes,
                use_transformer=not args.no_transformer,
                minimum_hold_steps=args.minimum_hold_steps,
            )
            env.reset(seed=args.seed + seed_offset)
            return monitor_class(
                env, filename=str(monitor_file) if monitor_file is not None else None
            )

        return factory

    train_env = dummy_vec_env(
        [make_env(0, args.out / "monitor" / "training_monitor.csv")]
    )
    eval_env = dummy_vec_env([make_env(20_000)])

    episode_callback = build_episode_callback(
        deps["BaseCallback"], args.out / "training_metrics.csv"
    )
    eval_callback = deps["EvalCallback"](
        eval_env,
        best_model_save_path=str(args.out / "best_model"),
        log_path=str(args.out / "eval_logs"),
        eval_freq=max(1, int(args.eval_freq)),
        n_eval_episodes=max(1, int(args.eval_episodes)),
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback = deps["CallbackList"]([episode_callback, eval_callback])

    model = deps["SAC"](
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        final_model_path = args.out / "sac_thermal_v4_final"
        model.save(str(final_model_path))
        episode_callback.save()
        plot_training(
            episode_callback.rows, args.out / "sac_thermal_v4_training_curve.png"
        )

        sb3_best = args.out / "best_model" / "best_model.zip"
        named_best = args.out / "sac_thermal_v4_best.zip"
        if sb3_best.exists():
            shutil.copy2(sb3_best, named_best)

        evaluation_model = model
        evaluation_model_name = "final"
        if sb3_best.exists():
            evaluation_model = deps["SAC"].load(
                str(sb3_best), device=args.device
            )
            evaluation_model_name = "best"
        evaluation = evaluate_model(
            model=evaluation_model,
            env_class=env_class,
            args=args,
            episodes=max(1, int(args.final_eval_episodes)),
        )
        evaluation["evaluated_model"] = evaluation_model_name
        _json_dump(args.out / "final_evaluation.json", evaluation)
        print("\nSAC 热惯性重训完成。")
        print(f"最终模型：{final_model_path}.zip")
        print(
            f"最佳模型：{named_best if named_best.exists() else '本轮未到首次评估步数'}"
        )
        print(f"逐回合指标：{args.out / 'training_metrics.csv'}")
        print(f"收敛图：{args.out / 'sac_thermal_v4_training_curve.png'}")
        print(f"最终评估：{args.out / 'final_evaluation.json'}")
    finally:
        train_env.close()
        eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
