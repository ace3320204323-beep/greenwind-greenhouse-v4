"""清洁能源/储能子系统回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from virtual_greenhouse.config import DEFAULT_DEVICES, EnergyParams
from virtual_greenhouse.energy_system import EnergySystem


def weather(solar=0.0, wind=0.0):
    return SimpleNamespace(
        outdoor_solar_irradiance_w_m2=solar,
        outdoor_wind_speed_m_s=wind,
    )


def actuators(heater=0.0, fan=0.0):
    return {
        "heater_01": heater,
        "fan_01": fan,
        "vent_01": 0.0,
        "curtain_01": 0.0,
        "irrigation_01": 0.0,
    }


class EnergySystemTest(unittest.TestCase):
    def setUp(self):
        self.params = EnergyParams()
        self.system = EnergySystem(self.params, list(DEFAULT_DEVICES), enabled=True)

    def test_solar_surplus_charges_battery(self):
        before = self.system.state.battery_soc_pct
        result = self.system.step(3600.0, 60.0, weather(solar=1000.0), actuators())
        self.assertLess(result.state.battery_power_kw, 0.0)
        self.assertGreater(result.state.battery_soc_pct, before)
        self.assertAlmostEqual(result.balance_residual_kw, 0.0, places=6)

    def test_outage_uses_battery_and_soc_falls(self):
        self.system.set_grid_status_override("OUTAGE")
        before = self.system.state.battery_soc_pct
        result = self.system.step(60.0, 60.0, weather(), actuators(heater=50.0))
        self.assertEqual(result.state.grid_power_kw, 0.0)
        self.assertGreater(result.state.battery_power_kw, 0.0)
        self.assertLess(result.state.battery_soc_pct, before)
        self.assertAlmostEqual(result.state.unserved_power_kw, 0.0, places=6)

    def test_low_soc_causes_real_load_shedding(self):
        self.system.set_grid_status_override("OUTAGE")
        self.system.state.battery_soc_pct = self.params.min_soc_pct
        result = self.system.step(60.0, 60.0, weather(), actuators(heater=100.0))
        self.assertGreater(result.state.unserved_power_kw, 0.0)
        self.assertLess(result.device_supply_ratio, 1.0)
        self.assertLess(result.effective_actuators["heater_01"], 100.0)

    def test_limited_grid_and_battery_close_power_balance(self):
        self.system.set_grid_status_override("LIMITED")
        result = self.system.step(60.0, 60.0, weather(), actuators(heater=100.0))
        self.assertLessEqual(result.state.grid_power_kw, self.params.grid_limited_import_kw)
        self.assertGreater(result.state.battery_power_kw, 0.0)
        self.assertAlmostEqual(result.balance_residual_kw, 0.0, places=6)

    def test_grid_status_auto_clears_manual_override(self):
        self.system.set_grid_status_override("OUTAGE")
        self.system.set_grid_status_override("AUTO")
        result = self.system.step(60.0, 60.0, weather(), actuators())
        self.assertEqual(result.state.grid_status, "NORMAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
