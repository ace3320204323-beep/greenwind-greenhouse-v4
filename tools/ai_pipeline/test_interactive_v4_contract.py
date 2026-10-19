from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tools" / "ai_pipeline" / "interactive_runtime_v2.py"
FRONTEND = ROOT / "tools" / "GreenWindConsoleV4.ps1"
CONFIG = ROOT / "config" / "ui_demo_config.json"


class InteractiveV4ContractTest(unittest.TestCase):
    def test_configuration_points_to_existing_artifacts(self) -> None:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key in (
            "scenario_path",
            "transformer_model_path",
            "transformer_meta_path",
            "sac_model_path",
        ):
            self.assertTrue((ROOT / data[key]).is_file(), key)

    def test_runtime_exposes_full_closed_loop_contract(self) -> None:
        source = RUNTIME.read_text(encoding="utf-8")
        for token in (
            '"environment"',
            '"prediction"',
            '"devices"',
            '"energy"',
            '"manual_targets_pct"',
            '"ai"',
            '"safety"',
            '"trend"',
        ):
            self.assertIn(token, source)

    def test_manual_and_ai_commands_are_supported(self) -> None:
        source = RUNTIME.read_text(encoding="utf-8")
        for command in (
            '"set_mode"',
            '"manual_targets"',
            '"preset"',
            '"ai_decision"',
            '"emergency_stop"',
            '"set_grid_status"',
            '"grid_event"',
        ):
            self.assertIn(command, source)

    def test_frontend_contains_required_pages_and_curves(self) -> None:
        source = FRONTEND.read_text(encoding="utf-8")
        for label in (
            "大棚实况",
            "控制中心",
            "AI智能托管",
            "能源协同",
            "实验与报告",
            "系统与投运",
            "实测棚温",
            "10分钟预测",
            "30分钟预测",
            "目标下限",
            "目标上限",
            "储能SOC",
            "清洁能源出力",
            "模拟市电停电",
            "已发送：",
            "已生效：",
        ):
            self.assertIn(label, source)

    def test_single_greenhouse_scope_is_explicit(self) -> None:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(data["source"], "virtual_greenhouse")
        self.assertIn(data["default_mode"], {"manual", "assist", "auto"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
