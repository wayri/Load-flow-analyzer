"""Small real CLI/receipt/AC integrations; image rendering is verified separately."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nrlarge.economics.__main__ import main
from nrlarge.economics.grid import check_grid
from nrlarge.economics.io import load_case, load_dispatch, save_case
from nrlarge.economics.model import Battery, Case, Interval, Site
from nrlarge.economics.optimizer import optimize
from nrlarge.economics.validation import validate_dispatch
from nrlarge.model import Case as GridCase, Load, Node, add_series, write_case


ROOT = Path(__file__).resolve().parents[2]


def periods(prices, load=1):
    start = datetime(2026,1,1,tzinfo=timezone.utc)
    return [Interval((start+timedelta(hours=k)).isoformat(),1,price,0,load_mw=load)
            for k,price in enumerate(prices)]


def simple_case():
    return Case([Site('meter',3,0)],{'meter':periods([10,100])},
                batteries=[Battery('B','meter',1,1,1,eta_charge=1,eta_discharge=1)])


def cli(arguments):
    stdout, stderr = io.StringIO(),io.StringIO()
    with redirect_stdout(stdout),redirect_stderr(stderr):
        code = main([str(v) for v in arguments])
    return code,stdout.getvalue(),stderr.getvalue()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'.task-check'; scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='economics-workflow-test-',dir=scratch)
        self.addCleanup(temporary.cleanup); self.directory = Path(temporary.name)
        # Exercise real schedule/cash CSV/JSON reports without repeatedly writing
        # chart previews on the critically full drive. Separate plot QA is real.
        self.plotter = self.enterContext(patch('nrlarge.economics.reports._plots',return_value={}))

    def input(self,case=None):
        case = simple_case() if case is None else case
        path = self.directory/'input'; save_case(case,path)
        return path

    def optimize_cli(self,path,out=None,*extra):
        out = self.directory/'result' if out is None else out
        code,stdout,stderr = cli(['optimize','--case',path,'--out',out,*extra])
        self.assertEqual(code,0,stderr)
        return out,json.loads(stdout)

    def test_cli_examples_for_all_roles_are_editable_valid_inputs(self):
        for role in ('battery','generator','discom','aggregator'):
            with self.subTest(role=role):
                destination = self.directory/role
                code,_,stderr = cli(['example','--role',role,'--out',destination])
                self.assertEqual(code,0,stderr)
                loaded = load_case(destination)
                self.assertEqual(loaded.role,role)
                self.assertEqual(len(next(iter(loaded.periods.values()))),24)
                if role == 'aggregator': self.assertGreater(len(loaded.sites),1)
                original = (destination/'case.json').read_bytes()
                code,_,_ = cli(['example','--role',role,'--out',destination])
                self.assertEqual(code,1)
                self.assertEqual((destination/'case.json').read_bytes(),original)

    def test_cli_optimize_baseline_saved_receipt_and_csv_cash_reconciliation(self):
        case = simple_case(); path = self.input(case)
        out,printed = self.optimize_cli(path,None,'--baseline')
        self.assertTrue(printed['feasible']); self.assertTrue(printed['optimal'])
        self.assertAlmostEqual(printed['total_cost'],20,delta=1e-6)
        self.assertEqual(load_case(out/'inputs'),case)
        dispatch = load_dispatch(out)
        self.assertTrue(validate_dispatch(case,dispatch)['valid'])
        summary = json.loads((out/'summary.json').read_text())
        self.assertAlmostEqual(summary['baseline']['total_cost'],110,delta=1e-6)
        self.assertAlmostEqual(summary['baseline']['baseline_cost_minus_dispatch_cost'],90,delta=1e-6)
        with (out/'cashflow.csv').open(newline='') as stream:
            interval_cost = sum(float(r['total_cost']) for r in csv.DictReader(stream))
        with (out/'horizon_charges.csv').open(newline='') as stream:
            charges = sum(float(r['amount']) for r in csv.DictReader(stream))
        self.assertAlmostEqual(interval_cost+charges,20,delta=1e-6)
        self.assertFalse((out/'INCOMPLETE').exists())
        code,stdout,stderr = cli(['check','--case',path,'--result',out])
        self.assertEqual(code,0,stderr); self.assertTrue(json.loads(stdout)['valid'])
        original = (out/'dispatch.json').read_bytes()
        code,_,_ = cli(['optimize','--case',path,'--out',out])
        self.assertEqual(code,1); self.assertEqual((out/'dispatch.json').read_bytes(),original)

    def test_check_rejects_saved_schedule_objective_and_accounting_tampering(self):
        path = self.input(); out,_ = self.optimize_cli(path)
        original = json.loads((out/'dispatch.json').read_text())
        for label in ('power','objective','line'):
            with self.subTest(tampering=label):
                altered = copy.deepcopy(original)
                if label == 'power': altered['site']['meter']['import_mw'][0] += .1
                elif label == 'objective': altered['objective_cost'] += 123
                else: altered['accounting']['lines'][0]['total_cost'] += 123
                receipt = self.directory/(label+'.json'); receipt.write_text(json.dumps(altered))
                code,stdout,stderr = cli(['check','--case',path,'--result',receipt])
                self.assertEqual(code,2,stderr)
                replay = json.loads(stdout)
                self.assertFalse(replay['valid']); self.assertTrue(replay['violations'])

    def test_cli_projected_state_replan_preserves_original_terminal_and_source_label(self):
        path = self.input(); out,_ = self.optimize_cli(path)
        state_path = self.directory/'state.json'
        code,_,stderr = cli(['state','--case',path,'--result',out,'--after-slot',1,'--out',state_path])
        self.assertEqual(code,0,stderr)
        state = json.loads(state_path.read_text())
        self.assertEqual(state['source'],'projected'); self.assertEqual(state['after_slot'],1)
        self.assertAlmostEqual(state['batteries']['B']['soc_mwh'],1,delta=1e-7)
        replanned = self.directory/'replan'
        code,stdout,stderr = cli(['replan','--case',path,'--state',state_path,'--out',replanned,'--baseline'])
        self.assertEqual(code,0,stderr); self.assertTrue(json.loads(stdout)['feasible'])
        inputs = load_case(replanned/'inputs')
        self.assertEqual(len(inputs.periods['meter']),1)
        self.assertEqual(inputs.batteries[0].initial_soc_mwh,1)
        self.assertEqual(inputs.batteries[0].terminal_soc_mwh,0)
        self.assertEqual(json.loads((replanned/'starting-state.json').read_text())['source'],'projected')
        dispatch = load_dispatch(replanned)
        self.assertAlmostEqual(dispatch.batteries['B']['energy_mwh'][-1],0,delta=1e-7)
        self.assertAlmostEqual(dispatch.accounting['total_cost'],0,delta=1e-6)
        original = state_path.read_bytes()
        code,_,_ = cli(['state','--case',path,'--result',out,'--after-slot',0,'--out',state_path])
        self.assertEqual(code,1); self.assertEqual(state_path.read_bytes(),original)

    def test_cli_terminal_boundary_cannot_replan_without_remaining_forecast(self):
        path = self.input(); out,_ = self.optimize_cli(path)
        state_path = self.directory/'closing.json'
        code,_,stderr = cli(['state','--case',path,'--result',out,'--after-slot',2,'--out',state_path])
        self.assertEqual(code,0,stderr)
        destination = self.directory/'closed-replan'
        code,_,stderr = cli(['replan','--case',path,'--state',state_path,'--out',destination])
        self.assertEqual(code,1); self.assertIn('time remaining',stderr)
        self.assertFalse(destination.exists())

    def test_cli_battery_candidate_planning_costs_and_ranking(self):
        path = self.input(); candidates = self.directory/'candidates.csv'
        with candidates.open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['candidate_id','battery_id','capacity_mwh','charge_max_mw','discharge_max_mw','horizon_fixed_cost'])
            writer.writerows([['none','B',0,0,0,0],['small','B',.5,.5,.5,5],['full','B',1,1,1,10]])
        out = self.directory/'planning'
        code,stdout,stderr = cli(['plan','--case',path,'--candidates',candidates,'--out',out])
        self.assertEqual(code,0,stderr)
        report = json.loads(stdout)
        self.assertEqual(report['best_candidate'],'full')
        self.assertEqual(report['ranking'],['full','small','none'])
        costs = {r['candidate_id']:r['total_horizon_cost'] for r in report['candidates']}
        for name,expected in (('none',110),('small',70),('full',30)):
            self.assertAlmostEqual(costs[name],expected,delta=1e-6)
        self.assertTrue(report['all_feasible_candidates_optimal'])
        with (out/'planning.csv').open(newline='') as stream: self.assertEqual(len(list(csv.DictReader(stream))),3)

    def test_cli_infeasibility_keeps_status_without_fake_cash_or_schedules(self):
        case = Case([Site('meter',0,0)],{'meter':periods([10],load=1)})
        path = self.input(case); out = self.directory/'infeasible'
        code,stdout,stderr = cli(['optimize','--case',path,'--out',out])
        self.assertEqual(code,2,stderr); self.assertFalse(json.loads(stdout)['feasible'])
        report = json.loads((out/'summary.json').read_text())
        self.assertFalse(report['feasible']); self.assertEqual(report['status'],'infeasible')
        for filename in ('dispatch.csv','cashflow.csv','battery_soc.csv'):
            self.assertFalse((out/filename).exists())
        self.plotter.assert_not_called()

    def grid_fixture(self,resistive=False,forecast_losses=0,rating=.6):
        nodes = [Node('source.P',phase='A',type='SLACK'),Node('source.N',phase='N',type='SLACK',vm=0),
                 Node('load.P',phase='A'),Node('load.N',phase='N',vm=0)]
        stamps = []
        impedances = [.02,.03] if resistive else [.02j,.03j]
        add_series(stamps,'line',['source.P','source.N'],['load.P','load.N'],np.diag(impedances))
        template = GridCase(nodes,stamps,base_mva=3)
        write_case(template,self.directory/'grid')
        rows = periods([20],load=.5); rows[0].loss_mw = forecast_losses
        case = Case([Site('meter',1,1)],{'meter':rows})
        dispatch = optimize(case)
        self.assertTrue(dispatch.feasible,dispatch.status)
        port = {'from':'load.P','to':'load.N','weight':1,'q_per_p':0}
        mapping = {'sites':{'meter':{'case':'grid','ports':{'fixed_load':[copy.deepcopy(port)],'renewable':[copy.deepcopy(port)]},
                                     'voltage_min_pu':.9,'voltage_max_pu':1.1,'interchange_tolerance_mw':1e-7,
                                     'terminal_current_limits_pu':{'line':rating}}}}
        return case,dispatch,mapping,template

    def test_lossless_ac_postcheck_balances_actual_neutral_and_slack_interchange(self):
        case,dispatch,mapping,_ = self.grid_fixture()
        report = check_grid(case,dispatch,mapping,self.directory)
        self.assertTrue(report['passed'],report)
        self.assertFalse(report['economic_dispatch_ac_constrained'])
        interval = report['intervals'][0]
        self.assertTrue(interval['converged']); self.assertTrue(interval['thermal_limits_checked'])
        self.assertAlmostEqual(interval['actual_pcc_mw'],.5,delta=1e-7)
        self.assertAlmostEqual(interval['scheduled_pcc_mw'],.5,delta=1e-7)
        self.assertLess(interval['voltage_min_pu'],1)
        self.assertGreater(interval['terminal_current_max_pu']['line'],.5)

    def test_resistive_ac_replay_uses_real_losses_and_configured_current_limits(self):
        # Exact two-wire DC-resistive AC special case: U=(1+sqrt(1-4RP))/2.
        resistance,power = .05,.5
        voltage = (1+math.sqrt(1-4*resistance*power))/2
        losses = resistance*(power/voltage)**2
        case,dispatch,mapping,_ = self.grid_fixture(True,losses)
        report = check_grid(case,dispatch,mapping,self.directory)
        self.assertTrue(report['passed'],report)
        entry = report['intervals'][0]
        self.assertAlmostEqual(entry['actual_pcc_mw'],power+losses,delta=1e-7)
        self.assertAlmostEqual(entry['voltage_min_pu'],voltage,delta=1e-7)
        mapping['sites']['meter']['terminal_current_limits_pu']['line'] = .4
        failed = check_grid(case,dispatch,mapping,self.directory)
        self.assertFalse(failed['passed'])
        self.assertTrue(failed['intervals'][0]['converged'])
        self.assertGreater(failed['intervals'][0]['terminal_current_max_pu']['line'],.4)

    def test_unforecast_resistive_losses_fail_postcheck_without_changing_dispatch(self):
        case,dispatch,mapping,_ = self.grid_fixture(True)
        before = copy.deepcopy(dispatch)
        report = check_grid(case,dispatch,mapping,self.directory)
        self.assertFalse(report['passed'])
        self.assertGreater(report['intervals'][0]['interchange_error_mw'],.01)
        self.assertEqual(dispatch,before)

    def test_ac_postcheck_maps_storage_charge_and_discharge_with_opposite_signs(self):
        _,_,mapping,_ = self.grid_fixture(rating=3)
        case = simple_case(); dispatch = optimize(case)
        mapping['sites']['meter']['ports']['battery:B'] = [
            {'from':'load.P','to':'load.N','weight':1,'q_per_p':0}]
        report = check_grid(case,dispatch,mapping,self.directory)
        self.assertTrue(report['passed'],report)
        self.assertAlmostEqual(report['intervals'][0]['actual_pcc_mw'],2,delta=1e-7)
        self.assertAlmostEqual(report['intervals'][1]['actual_pcc_mw'],0,delta=1e-7)
        del mapping['sites']['meter']['ports']['battery:B']
        with self.assertRaises(ValueError): check_grid(case,dispatch,mapping,self.directory)

    def test_grid_rejects_dirty_templates_missing_ports_bad_weights_and_ratings(self):
        case,dispatch,mapping,template = self.grid_fixture()
        for invalid in ('unmapped','weight','unknown-rating','zero-rating'):
            bad = copy.deepcopy(mapping); cfg = bad['sites']['meter']
            if invalid == 'unmapped': del cfg['ports']['renewable']
            elif invalid == 'weight': cfg['ports']['fixed_load'][0]['weight'] = .5
            elif invalid == 'unknown-rating': cfg['terminal_current_limits_pu'] = {'missing':1}
            else: cfg['terminal_current_limits_pu']['line'] = 0
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                check_grid(case,dispatch,bad,self.directory)
        template.loads.append(Load('existing','load.P','load.N',.1,.01))
        write_case(template,self.directory/'dirty')
        mapping['sites']['meter']['case'] = 'dirty'
        with self.assertRaises(ValueError): check_grid(case,dispatch,mapping,self.directory)


if __name__ == '__main__':
    unittest.main()
