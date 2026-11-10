"""Physical scenario ports and expected-outcome gates for the example collection."""
from copy import deepcopy
import unittest
import numpy as np
from benchmarks.run import studies, run_case, verify_sources
from nrlarge.model import Network, evaluate


class GalleryContracts(unittest.TestCase):
    def test_pinned_upstream_bytes(self):
        source = verify_sources()
        self.assertEqual(len(source['commit']),40)
        self.assertEqual(len([f for f in source['files'] if f['path'].endswith('.m')]),5)

    def test_rooftop_generation_has_an_explicit_neutral_return(self):
        study = next(s for s in studies() if s.name=='lv-rooftop-export')
        record,result = run_case(study)
        net = Network(study.case)
        _,_,current = evaluate(net,result.voltage,result.types,jacobian=False)
        pv = np.array([load.id.startswith('rooftop-') for load in study.case.loads])
        self.assertEqual(int(pv.sum()),3)
        self.assertTrue(all(load.target=='service.N' for load in study.case.loads if load.id.startswith('rooftop-')))
        self.assertTrue(all(node.p==0 for node in study.case.nodes))
        terminal_current = net.a[pv,:].T @ current[pv]
        self.assertAlmostEqual(abs(terminal_current.sum()),0,places=12)
        self.assertGreater(abs(terminal_current[net.index['service.N']]),1e-3)
        self.assertAlmostEqual(record['metrics']['embedded_generation_mw'],.0933333333333333,places=10)
        self.assertLess(record['metrics']['slack_mw'],0)
        self.assertLess(record['metrics']['power_balance_error_mva'],1e-9)

    def test_expected_failure_is_explicit_and_unexpected_success_is_rejected(self):
        source = studies()
        failed = next(s for s in source if s.name=='lv-neutral-extreme')
        self.assertFalse(failed.expected_converged)
        record,_ = run_case(failed)
        self.assertFalse(record['metrics']['converged'])
        wrong = deepcopy(next(s for s in source if s.name=='lv-base'))
        wrong.expected_converged = False
        with self.assertRaisesRegex(ArithmeticError,'expected converged'):
            run_case(wrong)

    def test_grounded_control_example_identifies_its_reference_and_hits_q_limits(self):
        study = next(s for s in studies() if s.name=='lv-reactive-controls')
        self.assertIn('ground',study.parameters['controller'])
        record,_ = run_case(study)
        self.assertTrue(record['metrics']['converged'])
        self.assertEqual(record['metrics']['pv_to_pq_conductors'],2)


if __name__ == '__main__': unittest.main()
