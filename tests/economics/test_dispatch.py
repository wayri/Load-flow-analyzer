"""Independent tiny dispatch references, physical replay and adversarial schedules."""
import copy
from datetime import datetime, timedelta, timezone
from itertools import product
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog

from nrlarge.economics.model import Battery, Case, FlexibleLoad, Generator, Interval, Site
from nrlarge.economics.optimizer import optimize
from nrlarge.economics.validation import validate_dispatch


def intervals(prices, duration=1, loads=None, sell=None, renewable=None):
    start = datetime(2026,1,1,tzinfo=timezone.utc)
    rows = []
    for k, price in enumerate(prices):
        rows.append(Interval((start+timedelta(hours=k*duration)).isoformat(),duration,price,
                             0 if sell is None else sell[k],
                             0 if loads is None else loads[k],
                             0 if renewable is None else renewable[k]))
    return rows


def battery_case(duration=1, degradation=0):
    return Case([Site('meter',3,2)],
                {'meter':intervals([10,100],duration,loads=[1,1])},
                batteries=[Battery('B','meter',1,1,1,eta_charge=.9,eta_discharge=.8,
                                   degradation_cost_per_mwh=degradation)])


def enumerate_two_slot_battery(case):
    """Independent exhaustive binary-mode reference with 16 continuous LPs.

    Variables are import/export/charge/discharge (two each), then E0/E1/E2.
    Each LP fixes both meter and battery binary modes; it does not use any
    optimizer model-building or accounting implementation.
    """
    site, battery = case.sites[0],case.batteries[0]
    periods = case.periods[site.id]
    assert len(periods) == 2 and len(case.sites) == len(case.batteries) == 1
    objective = np.zeros(11)
    for k,row in enumerate(periods):
        objective[k] = row.buy_price*row.duration_h
        objective[2+k] = -row.sell_price*row.duration_h
        objective[6+k] = battery.degradation_cost_per_mwh*row.duration_h
    equality, rhs = [],[]
    for k,row in enumerate(periods):
        balance = np.zeros(11)
        balance[k],balance[2+k],balance[4+k],balance[6+k] = 1,-1,-1,1
        equality.append(balance); rhs.append(row.load_mw+row.loss_mw)
        energy = np.zeros(11)
        energy[8+k+1],energy[8+k] = 1,-(1-battery.self_discharge_per_h)**row.duration_h
        energy[4+k] = -battery.eta_charge*row.duration_h
        energy[6+k] = row.duration_h/battery.eta_discharge
        equality.append(energy); rhs.append(0)
    best = math.inf
    high = battery.capacity_mwh if battery.soc_max_mwh is None else battery.soc_max_mwh
    terminal = battery.initial_soc_mwh if battery.terminal_soc_mwh is None else battery.terminal_soc_mwh
    for grid in product((0,1),repeat=2):
        for charging in product((0,1),repeat=2):
            bounds = ([(0,site.import_limit_mw*mode) for mode in grid]
                      +[(0,site.export_limit_mw*(1-mode)) for mode in grid]
                      +[(0,battery.charge_max_mw*mode) for mode in charging]
                      +[(0,battery.discharge_max_mw*(1-mode)) for mode in charging]
                      +[(battery.initial_soc_mwh,battery.initial_soc_mwh),
                        (battery.soc_min_mwh,high),(terminal,terminal)])
            result = linprog(objective,A_eq=np.array(equality),b_eq=np.array(rhs),bounds=bounds,method='highs')
            if result.success: best = min(best,float(result.fun))
    if not math.isfinite(best): raise AssertionError('Independent enumeration found no feasible schedule')
    return best


class DispatchTests(unittest.TestCase):
    def optimum(self,case):
        dispatch = optimize(case)
        self.assertTrue(dispatch.feasible,dispatch.status)
        self.assertTrue(dispatch.optimal,dispatch.status)
        self.assertEqual(dispatch.status,'optimal')
        replay = validate_dispatch(case,dispatch)
        self.assertTrue(replay['valid'],replay['violations'])
        self.assertAlmostEqual(dispatch.accounting['total_cost'],replay['accounting']['total_cost'],delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['net_profit'],replay['accounting']['net_profit'],delta=1e-6)
        return dispatch

    def test_two_slot_efficiency_reference_and_baseline_savings(self):
        dispatch = self.optimum(battery_case())
        battery = dispatch.batteries['B']; site = dispatch.site['meter']
        assert_allclose(battery['charge_mw'],[1,0],atol=1e-7)
        assert_allclose(battery['discharge_mw'],[0,.72],atol=1e-7)
        assert_allclose(battery['energy_mwh'],[0,.9,0],atol=1e-7)
        assert_allclose(site['import_mw'],[2,.28],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],48,delta=1e-6)
        baseline = copy.deepcopy(battery_case()); baseline.batteries = []
        no_battery = self.optimum(baseline)
        self.assertAlmostEqual(no_battery.accounting['total_cost'],110,delta=1e-6)
        self.assertAlmostEqual(no_battery.accounting['total_cost']-dispatch.accounting['total_cost'],62,delta=1e-6)

    def test_half_hour_dispatch_uses_mwh_not_mw_as_energy(self):
        dispatch = self.optimum(battery_case(duration=.5))
        assert_allclose(dispatch.batteries['B']['energy_mwh'],[0,.45,0],atol=1e-7)
        assert_allclose(dispatch.batteries['B']['discharge_mw'],[0,.72],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],24,delta=1e-6)

    def test_delivered_ac_discharge_degradation_cost_and_throughput_budget(self):
        moderate = self.optimum(battery_case(degradation=10))
        self.assertAlmostEqual(moderate.accounting['total_cost'],55.2,delta=1e-6)
        expensive = self.optimum(battery_case(degradation=1000))
        assert_allclose(expensive.batteries['B']['charge_mw'],0,atol=1e-7)
        assert_allclose(expensive.batteries['B']['discharge_mw'],0,atol=1e-7)
        self.assertAlmostEqual(expensive.accounting['total_cost'],110,delta=1e-6)
        case = battery_case(); case.periods['meter'][0].buy_price = 1
        case.batteries[0].eta_charge = case.batteries[0].eta_discharge = 1
        case.batteries[0].max_discharge_mwh = .25
        capped = self.optimum(case)
        assert_allclose(capped.batteries['B']['discharge_mw'],[0,.25],atol=1e-7)
        self.assertAlmostEqual(capped.accounting['total_cost'],76.25,delta=1e-6)

    def test_explicit_terminal_soc_and_fractional_self_discharge(self):
        case = Case([Site('meter',2,2)],{'meter':intervals([20],duration=.5)},
                    batteries=[Battery('B','meter',1,1,1,eta_charge=1,eta_discharge=1,
                                       initial_soc_mwh=1,terminal_soc_mwh=1,self_discharge_per_h=.1)])
        dispatch = self.optimum(case)
        required_energy = 1-math.sqrt(.9)
        self.assertAlmostEqual(dispatch.batteries['B']['charge_mw'][0],required_energy/.5,delta=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],20*required_energy,delta=1e-6)
        case.batteries[0].initial_soc_mwh = 0
        case.batteries[0].terminal_soc_mwh = .5
        case.batteries[0].self_discharge_per_h = 0
        dispatch = self.optimum(case)
        assert_allclose(dispatch.batteries['B']['energy_mwh'],[0,.5],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],10,delta=1e-6)

    def test_negative_prices_do_not_create_same_slot_energy_loops(self):
        case = Case([Site('meter',2,2)],{'meter':intervals([-10],sell=[20])},
                    batteries=[Battery('B','meter',1,1,1,eta_charge=1,eta_discharge=1)])
        dispatch = self.optimum(case)
        for key in ('import_mw','export_mw'): assert_allclose(dispatch.site['meter'][key],0,atol=1e-7)
        for key in ('charge_mw','discharge_mw'): assert_allclose(dispatch.batteries['B'][key],0,atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],0,delta=1e-7)

    def test_distinct_meters_can_import_and_export_at_the_same_time(self):
        a = intervals([50],loads=[1]); a[0].loss_mw = .1; a[0].retail_price = 80
        b = intervals([50],sell=[30],renewable=[1])
        case = Case([Site('A',2,2),Site('B',2,2)],{'A':a,'B':b},role='aggregator')
        dispatch = self.optimum(case)
        assert_allclose(dispatch.site['A']['import_mw'],[1.1],atol=1e-7)
        assert_allclose(dispatch.site['B']['export_mw'],[1],atol=1e-7)
        assert_allclose(dispatch.site['B']['renewable_used_mw'],[1],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['import_cost'],55,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['export_revenue'],30,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['retail_revenue'],80,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],-55,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['net_profit'],55,delta=1e-6)

    def test_incremental_demand_charge_uses_historical_peak_once(self):
        case = Case([Site('meter',3,0,demand_charge_per_mw=10,historical_peak_mw=1)],
                    {'meter':intervals([20],loads=[2])})
        dispatch = self.optimum(case)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],50,delta=1e-6)
        case.sites[0].historical_peak_mw = 3
        dispatch = self.optimum(case)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],40,delta=1e-6)

    def test_generator_startup_and_ramps_have_a_hand_computed_cost(self):
        case = Case([Site('meter',2,0)],{'meter':intervals([100,100],loads=[.6,.6])},
                    generators=[Generator('G','meter',1,40,p_min_mw=.2,no_load_cost_per_h=3,
                                          startup_cost=7,ramp_up_mw_per_h=.1,ramp_down_mw_per_h=.1,
                                          startup_ramp_mw=.4,shutdown_ramp_mw=.4)])
        dispatch = self.optimum(case)
        assert_allclose(dispatch.generators['G']['p_mw'],[.4,.5],atol=1e-7)
        assert_allclose(dispatch.generators['G']['on'],[1,1],atol=1e-7)
        assert_allclose(dispatch.generators['G']['startup'],[1,0],atol=1e-7)
        assert_allclose(dispatch.site['meter']['import_mw'],[.2,.1],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],79,delta=1e-6)

    def test_minimum_up_time_and_initial_minimum_down_obligation(self):
        case = Case([Site('meter',2,0)],{'meter':intervals([5,100,5],loads=[.5,.5,0])},
                    generators=[Generator('G','meter',1,40,p_min_mw=.5,no_load_cost_per_h=2,
                                          min_up_h=2,initial_state_h=10)])
        dispatch = self.optimum(case)
        assert_allclose(dispatch.generators['G']['on'],[1,1,0],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],44,delta=1e-6)
        case = Case([Site('meter',2,0)],{'meter':intervals([100,100],loads=[.5,.5])},
                    generators=[Generator('G','meter',1,0,min_down_h=2,initial_state_h=0)])
        dispatch = self.optimum(case)
        assert_allclose(dispatch.generators['G']['on'],[0,0],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],100,delta=1e-6)

    def test_flexible_load_exact_energy_window_and_value(self):
        case = Case([Site('meter',2,0)],{'meter':intervals([0,50,10],duration=.5)},
                    flexible_loads=[FlexibleLoad('F','meter',.75,1,start_slot=1,end_slot=3,value_per_mwh=100)])
        dispatch = self.optimum(case)
        assert_allclose(dispatch.flexible['F']['p_mw'],[0,.5,1],atol=1e-7)
        self.assertAlmostEqual(dispatch.accounting['import_cost'],17.5,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['flexible_revenue'],75,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],-57.5,delta=1e-6)
        self.assertAlmostEqual(dispatch.accounting['net_profit'],57.5,delta=1e-6)

    def test_exhaustive_binary_mode_lp_reference_matches_milp(self):
        case = Case([Site('meter',2,2)],{'meter':intervals([-5,30],loads=[.2,.8],sell=[10,20])},
                    batteries=[Battery('B','meter',.6,.6,.6,eta_charge=.9,eta_discharge=.8,
                                       degradation_cost_per_mwh=2)])
        independent = enumerate_two_slot_battery(case)
        self.assertAlmostEqual(independent,7.904,delta=1e-7)
        dispatch = self.optimum(case)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],independent,delta=1e-6)

    def test_infeasible_demand_is_reported_without_fabricated_schedule(self):
        case = Case([Site('meter',0,0)],{'meter':intervals([10],loads=[1])})
        dispatch = optimize(case)
        self.assertFalse(dispatch.feasible)
        self.assertFalse(dispatch.optimal)
        self.assertEqual(dispatch.status,'infeasible')

    def test_replay_rejects_balance_soc_binary_and_generator_transition_tampering(self):
        case = battery_case(); dispatch = self.optimum(case)
        corrupt = copy.deepcopy(dispatch); corrupt.site['meter']['import_mw'][0] += .05
        self.assertFalse(validate_dispatch(case,corrupt)['valid'])
        corrupt = copy.deepcopy(dispatch); corrupt.batteries['B']['energy_mwh'][1] += .05
        self.assertFalse(validate_dispatch(case,corrupt)['valid'])
        corrupt = copy.deepcopy(dispatch); corrupt.site['meter']['grid_mode'][0] = .5
        self.assertFalse(validate_dispatch(case,corrupt)['valid'])
        case = Case([Site('meter',2,0)],{'meter':intervals([100,100],loads=[.5,.5])},
                    generators=[Generator('G','meter',1,10,startup_cost=3)])
        dispatch = self.optimum(case)
        corrupt = copy.deepcopy(dispatch); corrupt.generators['G']['startup'][0] = 0
        self.assertFalse(validate_dispatch(case,corrupt)['valid'])

    def test_bad_inputs_reject_noncontiguous_times_efficiency_and_unknown_site(self):
        case = battery_case(); case.periods['meter'][1].time = '2026-01-01T03:00:00+00:00'
        with self.assertRaises(ValueError): optimize(case)
        case = battery_case(); case.batteries[0].eta_charge = 1.1
        with self.assertRaises(ValueError): optimize(case)
        case = battery_case(); case.batteries[0].site = 'missing'
        with self.assertRaises(ValueError): optimize(case)


if __name__ == '__main__':
    unittest.main()
