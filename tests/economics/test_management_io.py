"""Editable input contracts and explicit state carry across forecast boundaries."""
from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from nrlarge.economics.io import load_case, save_case, load_dispatch, save_dispatch
from nrlarge.economics.management import baseline_case, replan, state_at
from nrlarge.economics.model import Battery, Case, FlexibleLoad, Generator, Interval, Site
from nrlarge.economics.optimizer import optimize


def managed_case():
    return Case([Site("S", 5, 5, 10, .1)], {"S": [
        Interval("2026-11-11T00:00:00+00:00", 1, 10, 10, 1),
        Interval("2026-11-11T01:00:00+00:00", 1, 100, 100, 1)]},
        [Battery("B", "S", 1, 1, 1, eta_charge=.9, eta_discharge=.8,
                 max_discharge_mwh=1)],
        [Generator("G", "S", .5, 1000, p_min_mw=.2, initial_on=1,
                   initial_p_mw=.2, min_up_h=3, initial_state_h=1)],
        [FlexibleLoad("F", "S", .5, .5)])


def replace_table(path, edit):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    edit(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class Inputs(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2]/".task-check"
        scratch.mkdir(exist_ok=True)
        self.owned = tempfile.TemporaryDirectory(prefix="economics-io-", dir=scratch)
        self.addCleanup(self.owned.cleanup)
        self.directory = Path(self.owned.name)

    def test_json_csv_roundtrip_and_assets_are_real_editable_overrides(self):
        case = managed_case(); case.flexible_loads[0].end_slot = 1
        directory = self.directory/"inputs"
        save_case(case, directory)
        self.assertEqual(asdict(load_case(directory)), asdict(case))
        replace_table(directory/"periods.csv", lambda rows: rows[0].update(buy_price="-25"))
        replace_table(directory/"batteries.csv", lambda rows: rows[0].update(charge_max_mw=".5"))
        changed = load_case(directory)
        self.assertEqual(changed.periods["S"][0].buy_price, -25)
        self.assertEqual(changed.batteries[0].charge_max_mw, .5)
        # Header-only battery CSV explicitly removes batteries from JSON input.
        replace_table(directory/"batteries.csv", lambda rows: rows.clear())
        self.assertEqual(load_case(directory).batteries, [])

    def test_unknown_fields_and_unknown_site_ids_are_not_silently_ignored(self):
        directory = self.directory/"inputs"; save_case(managed_case(), directory)
        path = directory/"case.json"
        original = json.loads(path.read_text())
        bad = deepcopy(original); bad["batteries"][0]["soc_typo"] = 1
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ValueError): load_case(directory)
        path.write_text(json.dumps(original), encoding="utf-8")
        replace_table(directory/"periods.csv", lambda rows: rows[0].update(site="unknown"))
        with self.assertRaises(ValueError): load_case(directory)

    def test_duplicate_json_nonfinite_values_and_bad_csv_columns_are_rejected(self):
        directory = self.directory/"inputs"; save_case(managed_case(), directory)
        path = directory/"case.json"; original = path.read_text()
        for value in ('{"currency":"CUR","currency":"USD"}',
                      original.replace('"fixed_horizon_cost": 0', '"fixed_horizon_cost": NaN')):
            path.write_text(value, encoding="utf-8")
            with self.assertRaises(ValueError): load_case(directory)
        path.write_text(original, encoding="utf-8")
        csv_path = directory/"sites.csv"
        csv_path.write_text(csv_path.read_text().replace("id,", "unknown,"), encoding="utf-8")
        with self.assertRaises(ValueError): load_case(directory)

    def test_existing_files_are_preserved_and_dispatch_receipt_roundtrips(self):
        directory = self.directory/"inputs"; save_case(managed_case(), directory)
        before = (directory/"case.json").read_bytes()
        with self.assertRaises(ValueError): save_case(managed_case(), directory)
        self.assertEqual((directory/"case.json").read_bytes(), before)
        result = optimize(managed_case()); self.assertTrue(result.feasible)
        path = save_dispatch(result, self.directory/"schedule.json")
        loaded = load_dispatch(path)
        self.assertEqual(loaded.site, result.site)
        self.assertEqual(loaded.batteries, result.batteries)
        self.assertAlmostEqual(loaded.objective_cost, result.objective_cost)
        with self.assertRaises(ValueError): save_dispatch(result, path)


class RollingState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = managed_case()
        cls.dispatch = optimize(cls.case)
        if not cls.dispatch.feasible:
            raise AssertionError(cls.dispatch.validation)

    def test_opening_closing_and_all_managed_obligations_are_carried(self):
        initial = state_at(self.case, self.dispatch, 0)
        self.assertEqual(initial["source"], "projected")
        self.assertEqual(initial["batteries"]["B"]["soc_mwh"], 0)
        self.assertEqual(initial["flexible_loads"]["F"]["remaining_energy_mwh"], .5)
        midway = state_at(self.case, self.dispatch, 1)
        continued = replan(self.case, midway, midway["boundary"])
        self.assertAlmostEqual(continued.batteries[0].initial_soc_mwh, .9)
        self.assertEqual(continued.batteries[0].terminal_soc_mwh, 0,
                         "Default terminal SOC must retain original horizon commitment")
        self.assertEqual(continued.generators[0].initial_on, 1)
        self.assertAlmostEqual(continued.generators[0].initial_p_mw, .2)
        self.assertAlmostEqual(continued.generators[0].initial_state_h, 2)
        self.assertEqual(continued.flexible_loads[0].energy_mwh, 0)
        self.assertGreater(continued.sites[0].historical_peak_mw, .1)
        self.assertTrue(optimize(continued).feasible)
        closing = state_at(self.case, self.dispatch, 2)
        self.assertAlmostEqual(closing["batteries"]["B"]["remaining_discharge_mwh"], .28)
        self.assertAlmostEqual(closing["generators"]["G"]["status_duration_h"], 3)
        with self.assertRaises(ValueError): replan(self.case, closing, closing["boundary"])

    def test_missing_asset_states_impossible_values_and_boundary_mismatch_fail(self):
        state = state_at(self.case, self.dispatch, 1)
        for group, asset in (("batteries", "B"), ("generators", "G"),
                             ("sites", "S"), ("flexible_loads", "F")):
            invalid = deepcopy(state); del invalid[group][asset]
            with self.subTest(group=group), self.assertRaises(ValueError):
                replan(self.case, invalid, state["boundary"])
        changes = [("batteries", "B", "soc_mwh", 99),
                   ("batteries", "B", "remaining_discharge_mwh", 99),
                   ("generators", "G", "on", 0),
                   ("generators", "G", "status_duration_h", -1),
                   ("sites", "S", "historical_peak_mw", 0),
                   ("flexible_loads", "F", "remaining_energy_mwh", .6)]
        for group, asset, field, value in changes:
            invalid = deepcopy(state); invalid[group][asset][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                replan(self.case, invalid, state["boundary"])
        with self.assertRaises(ValueError): replan(self.case, state, self.case.periods["S"][0].time)
        invalid = deepcopy(state); invalid["unrecognized"] = 0
        with self.assertRaises(ValueError): replan(self.case, invalid, state["boundary"])

    def test_replacement_forecast_and_measured_source_preserve_exact_boundary(self):
        state = state_at(self.case, self.dispatch, 1)
        state["source"] = "measured"
        replacement = deepcopy(self.case)
        replacement.periods["S"] = replacement.periods["S"][1:]
        continued = replan(replacement, state, state["boundary"])
        self.assertEqual(len(continued.periods["S"]), 1)
        self.assertAlmostEqual(continued.batteries[0].initial_soc_mwh, .9)
        self.assertEqual(continued.generators[0].initial_state_h, 2)
        self.assertEqual(state["source"], "measured")

    def test_no_battery_baseline_does_not_change_other_assets_or_original_case(self):
        without = baseline_case(self.case)
        self.assertEqual(without.batteries, [])
        self.assertEqual(asdict(without.generators[0]), asdict(self.case.generators[0]))
        self.assertEqual(asdict(without.flexible_loads[0]), asdict(self.case.flexible_loads[0]))
        self.assertEqual(without.periods, self.case.periods)
        without.generators[0].initial_p_mw = .4
        self.assertEqual(self.case.generators[0].initial_p_mw, .2)
        with self.assertRaises(ValueError): state_at(self.case, self.dispatch, -1)
        with self.assertRaises(ValueError): state_at(self.case, self.dispatch, True)


if __name__ == "__main__":
    unittest.main()
