"""热惯性模型回归测试。"""

from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from virtual_greenhouse.config import load_config
from virtual_greenhouse.greenhouse import GreenhousePlant, ZONES


BASE = Path(__file__).resolve().parent


def weather(temp=0.0):
    return SimpleNamespace(
        outdoor_air_temperature_c=temp,
        outdoor_relative_humidity_pct=60.0,
        outdoor_solar_irradiance_w_m2=0.0,
        outdoor_wind_speed_m_s=0.0,
    )


def actuators(heater=0.0):
    return {
        "heater_01": heater, "vent_01": 0.0, "fan_01": 0.0,
        "curtain_01": 100.0, "irrigation_01": 0.0,
    }


def initialise(plant, temp=8.0):
    for zid in ZONES:
        plant.state.air_temp[zid] = temp
        plant.state.air_abs_humidity[zid] = 5.0
    plant.state.canopy_temp = temp
    plant.state.substrate_temp = temp
    plant.state.structure_temp = temp
    plant.state.cover_temp = temp


class ThermalInertiaTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(BASE / "scenarios" / "shandong_winter_strawberry.json")

    def test_configured_capacity_is_used(self):
        plant = GreenhousePlant(self.cfg)
        expected = (self.cfg.zones["CENTER"].air_heat_capacity_kj_k
                    * self.cfg.zones["CENTER"].thermal_inertia)
        self.assertAlmostEqual(plant._zone_heat_capacity(ZONES[1]), expected)

    def test_heating_response_is_gradual(self):
        plant = GreenhousePlant(self.cfg)
        initialise(plant)
        before = plant.state.air_temp["CENTER"]
        plant.step(60.0, 60.0, weather(), actuators(100.0), random.Random(42))
        change = plant.state.air_temp["CENTER"] - before
        self.assertGreater(change, 0.0)
        self.assertLess(change, 1.0, "单步温度变化过大，热惯性可能未生效")

    def test_larger_thermal_mass_responds_more_slowly(self):
        normal = GreenhousePlant(self.cfg)
        low_cfg = copy.deepcopy(self.cfg)
        for zone in low_cfg.zones.values():
            zone.air_heat_capacity_kj_k *= 0.35
        low_cfg.plant.canopy_heat_capacity_kj_k *= 0.35
        low_cfg.plant.substrate_heat_capacity_kj_k *= 0.35
        low_cfg.plant.structure_heat_capacity_kj_k *= 0.35
        low = GreenhousePlant(low_cfg)
        initialise(normal)
        initialise(low)
        rng1, rng2 = random.Random(42), random.Random(42)
        for step in range(30):
            normal.step(step * 60.0, 60.0, weather(), actuators(100.0), rng1)
            low.step(step * 60.0, 60.0, weather(), actuators(100.0), rng2)
        self.assertGreater(low.state.air_temp["CENTER"], normal.state.air_temp["CENTER"])

    def test_equal_temperature_has_no_internal_heat_exchange_drift(self):
        plant = GreenhousePlant(self.cfg)
        initialise(plant, temp=10.0)
        before = dict(plant.state.air_temp)
        plant.step(60.0, 60.0, weather(temp=10.0), actuators(0.0), random.Random(42))
        for zid in ZONES:
            self.assertAlmostEqual(plant.state.air_temp[zid], before[zid], places=6)
        self.assertAlmostEqual(plant.state.canopy_temp, 10.0, places=6)
        self.assertAlmostEqual(plant.state.substrate_temp, 10.0, places=6)
        self.assertAlmostEqual(plant.state.structure_temp, 10.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
