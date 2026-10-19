"""虚拟大棚命令行入口。

用法：
    python -m tools.virtual_greenhouse.run --config <场景json> [--seed N] [--out DIR]

直接运行固定 seed 的完整仿真，输出 runs/{run_id}/ 目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许作为脚本直接运行（python tools/virtual_greenhouse/run.py）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from virtual_greenhouse.config import load_config
    from virtual_greenhouse.runner import VirtualGreenhouse
else:
    from .config import load_config
    from .runner import VirtualGreenhouse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="绿风润耕虚拟大棚仿真")
    parser.add_argument("--config", default=None,
                        help="场景 JSON 路径（缺省用山东冬季草莓 72h）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", default=None, help="输出目录（缺省 runs/{run_id}）")
    parser.add_argument("--evaluate", action="store_true", help="评价模式：返回真值")
    args = parser.parse_args(argv)

    base = Path(__file__).resolve().parent
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = base / "scenarios" / "shandong_winter_strawberry.json"

    cfg = load_config(cfg_path)
    print(f"场景   : {cfg.scenario_id}")
    print(f"时长   : {cfg.simulation.duration_hours} h, dt={cfg.simulation.dt_seconds}s")
    print(f"种子   : {args.seed}")
    print(f"阶段   : {cfg.crop_stage.value}")

    out_dir = Path(args.out) if args.out else None
    vg = VirtualGreenhouse(cfg, args.seed, evaluate=args.evaluate, output_dir=out_dir)

    obs0 = vg.reset()
    print(f"初始观测: {len(obs0.observations)} 个测点")

    total_steps = int(cfg.simulation.duration_hours * 3600 / cfg.simulation.dt_seconds)
    done = False
    step = 0
    while not done and step < total_steps:
        # P0 阶段：无控制器，命令为空，纯物理演算（供验收物理模型）
        result = vg.step(None)
        done = result.done
        step += 1
        if step % 240 == 0:
            # 每 4 小时打印一次进度
            print(f"  进度 {step}/{total_steps} 步, sim={vg.sim_seconds/3600:.1f}h")

    summary = vg.finalize()
    metrics = summary["metrics"]["environment"]
    print("\n仿真完成")
    print(f"  run_id     : {summary['manifest']['run_id']}")
    print(f"  温度达标率 : {metrics['temperature_in_range_ratio']}")
    print(f"  温度 MAE   : {metrics['temperature_mae_c']} ℃")
    print(f"  湿度达标率 : {metrics['humidity_in_range_ratio']}")
    if vg.recorder:
        print(f"  输出目录   : {vg.recorder.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
