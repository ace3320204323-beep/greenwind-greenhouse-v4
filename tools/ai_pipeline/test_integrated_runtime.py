"""统一运行时的轻量测试，不加载 Torch、Gymnasium 或 SB3。"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from integrated_runtime_v1 import (
    IntegratedRuntime,
    HumanReviewBroker,
    RuntimeConfig,
    normalized_to_strategy,
    strategy_to_normalized,
)


class FakePolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del observation, deterministic
        return np.zeros(4, dtype=np.float32), None


class FakeEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.closed = False
        self._latest_record = {}

    def _info(self) -> dict:
        temperature = 9.0 + 0.1 * self.step_count
        self._latest_record = {
            "timestamp": f"2026-01-01T00:{self.step_count:02d}:00",
            "out_temp_c": 1.0,
            "out_rh_pct": 60.0,
            "out_solar_w_m2": 0.0,
            "out_wind_m_s": 2.0,
        }
        command = {name: 10.0 for name in ("heater", "vent", "curtain", "fan")}
        return {
            "sim_hour": self.step_count / 12.0,
            "temperature_c": temperature,
            "relative_humidity_pct": 75.0,
            "target_band_c": (8.0, 10.0),
            "pred_10min_c": temperature + 0.1,
            "pred_30min_c": temperature + 0.2,
            "transformer_status": "READY",
            "transformer_quality": "GOOD",
            "strategy_action": normalized_to_strategy(np.zeros(4)),
            "requested_targets_pct": command,
            "safe_targets_pct": command,
            "actuals_pct": command,
            "safety_overrides": (),
            "planner_constraint_events": (),
            "planner_objective": 0.25,
        }

    def reset(self, seed: int, options: dict):
        del seed, options
        return np.zeros(28, dtype=np.float32), self._info()

    def step(self, action: np.ndarray):
        self.step_count += 1
        return np.zeros(28, dtype=np.float32), -0.1, False, self.step_count >= 3, self._info()

    def close(self) -> None:
        self.closed = True


class RuntimeTest(unittest.TestCase):
    def test_packaged_transformer_contract(self) -> None:
        ai_dir = Path(__file__).resolve().parent
        meta = json.loads(
            (ai_dir / "outputs_v2_1_residual" / "transformer_v2_meta.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(meta["window_minutes"], 60)
        self.assertEqual(meta["horizons_minutes"], [10, 30])
        self.assertEqual(meta["target_representation"], "future_temperature_minus_current_temperature")
        self.assertEqual(len(meta["features"]), 31)
        v3_source = (ai_dir / "greenhouse_sac_env_v3.py").read_text(encoding="utf-8")
        self.assertIn("GreenhouseSACEnv.OBSERVATION_NAMES +", v3_source)
        self.assertIn("StrategyToControlV3", v3_source)

    def test_strategy_round_trip(self) -> None:
        strategy = {
            "temperature_offset_c": 1.0,
            "comfort_weight": 0.8,
            "energy_weight": 0.25,
            "dehumidify_priority": 0.9,
        }
        restored = normalized_to_strategy(strategy_to_normalized(strategy))
        for key, value in strategy.items():
            self.assertAlmostEqual(restored[key], value, places=5)

    def test_integrated_run_writes_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(
                mode="auto",
                max_control_steps=2,
                output_root=Path(directory),
                print_progress=False,
            )
            fake = FakeEnv()
            runtime = IntegratedRuntime(
                config,
                env_factory=lambda _: fake,
                policy_loader=lambda _: FakePolicy(),
            )
            summary = runtime.run()
            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["ended_reason"], "max_control_steps")
            self.assertTrue(fake.closed)
            expected = [
                "state_trace.csv", "prediction_trace.csv", "strategy_trace.csv",
                "command_trace.csv", "safety_trace.csv", "latest_state.json",
                "runtime_summary.json", "runtime_manifest.json", "runtime_events.jsonl",
            ]
            for name in expected:
                self.assertTrue((runtime.run_dir / name).is_file(), name)
            latest = json.loads((runtime.run_dir / "latest_state.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["runtime"]["control_step"], 2)

    def test_stop_is_idempotent(self) -> None:
        runtime = IntegratedRuntime(RuntimeConfig(), env_factory=lambda _: FakeEnv())
        self.assertTrue(runtime.request_stop())
        self.assertFalse(runtime.request_stop())

    def test_human_review_approve_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broker = HumanReviewBroker(root, timeout_seconds=2.0, poll_seconds=0.01)

            def operator() -> None:
                for _ in range(100):
                    if broker.pending_file.is_file():
                        pending = json.loads(broker.pending_file.read_text(encoding="utf-8"))
                        response = {
                            "decision_id": pending["decision_id"],
                            "decision": "approve",
                            "operator": "TEST_OPERATOR",
                        }
                        path = broker.response_dir / f"{pending['decision_id']}.json"
                        path.write_text(json.dumps(response), encoding="utf-8")
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=operator)
            thread.start()
            recommendation = np.asarray([0.2, 0.3, -0.1, 0.5], dtype=np.float32)
            applied, policy, review = broker.request(
                step=0,
                info=FakeEnv()._info(),
                recommended_action=recommendation,
                fixed_action=np.zeros(4, dtype=np.float32),
                stop_event=threading.Event(),
            )
            thread.join(timeout=1.0)
            np.testing.assert_allclose(applied, recommendation)
            self.assertEqual(policy, "human_approved_sac")
            self.assertEqual(review["operator"], "TEST_OPERATOR")
            self.assertFalse(broker.pending_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
