"""Static source safety, phase import contracts and independent polar references."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from benchmarks.cases import Dataset, independent_reference, load_matpower, to_phase_case
from nrlarge.model import Network, evaluate
from nrlarge.solver import solve


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT/'benchmarks'/'sources'/'matpower'


def bus(identifier, kind, pd=0, qd=0, gs=0, bs=0, vm=1, va=0):
    return [identifier,kind,pd,qd,gs,bs,1,vm,va,110,1,1.1,.9]


def generator(identifier, pg=0, qg=0, qmax=100, qmin=-100, vg=1, status=1):
    return [identifier,pg,qg,qmax,qmin,vg,100,status,200,0]+[0]*11


def branch(a,b,r=0,x=.1,charging=0,tap=0,shift=0,status=1):
    return [a,b,r,x,charging,100,100,100,tap,shift,status,-360,360]


def analytic_case():
    return Dataset('analytic_two_bus',100,np.array([bus(1,3),bus(9,1,50,45)],float),
                   np.array([generator(1)],float),np.array([branch(1,9)],float))


def feature_case():
    return Dataset('feature_case',100,
                   np.array([bus(10,3,vm=.97,va=3),bus(40,2,8,2,1,-2,vm=.98),
                             bus(700,1,15,5,2,3)],float),
                   np.array([generator(10,60,vg=1.04),generator(40,20,qmax=30,qmin=-10,vg=1.02),
                             generator(40,10,qmax=40,qmin=-20,vg=1.02),
                             generator(40,999,999,vg=1.8,status=0),
                             generator(700,4,2,qmax=0,qmin=0)],float),
                   np.array([branch(10,40,.02,.1,.04,1.03,5),branch(40,700,.015,.08,.02),
                             branch(10,700,x=0,status=0)],float))


def source_text(data):
    def matrix(name,values):
        return 'mpc.'+name+' = [\n'+'\n'.join(' '.join(f'{v:.17g}' for v in row)+';' for row in values)+'\n];\n'
    return ('function mpc = '+data.name+'\n% Comment is never executed.\n'
            "mpc.version = '2';\nmpc.baseMVA = "+str(data.base_mva)+';\n'
            +matrix('bus',data.bus)+matrix('gen',data.gen)+matrix('branch',data.branch)
            +"mpc.gencost = [2 0 0 2 1 0];\nmpc.bus_name = {'A % quoted'; 'B''s bus'};\n")


class BenchmarkCaseTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'.task-check'; scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='benchmark-case-test-',dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def parse_text(self,text):
        path = self.directory/'case.m'; path.write_text(text,encoding='utf-8')
        return load_matpower(path)

    def test_static_parser_preserves_data_comments_quotes_and_exponents(self):
        case = feature_case()
        loaded = self.parse_text(source_text(case).replace('baseMVA = 100;', 'baseMVA = 1D2;'))
        self.assertEqual(loaded.name,case.name); self.assertEqual(loaded.base_mva,100)
        for name in ('bus','gen','branch'): assert_allclose(getattr(loaded,name),getattr(case,name),atol=0)

    def test_parser_rejects_executable_statements_expressions_and_invalid_shapes(self):
        original = source_text(analytic_case())
        unsupported = [original+"system('untrusted');\n",original+"mpc.bus(:,3) = 0;\n",
                       original+"mpc.extra = 7;\n", original+"mpc.baseMVA = 200;\n",
                       original.replace('baseMVA = 100;', 'baseMVA = 50*2;'),
                       original.replace("version = '2'", "version = '1'"),
                       original.replace('function mpc = analytic_two_bus', 'function result = run_this'),
                       original.replace('mpc.bus = [', 'mpc.bus = [sin(0);'),
                       original.replace('mpc.branch = [', 'mpc.branch = [1 2;'),
                       original.replace('baseMVA = 100;', 'baseMVA = 1e309;')]
        for text in unsupported:
            with self.subTest(source=text[-50:]):
                with self.assertRaises(ValueError): self.parse_text(text)

    def test_phase_import_shunts_transformers_statuses_and_multiple_generators(self):
        data = feature_case(); case = to_phase_case(data); net = Network(case)
        self.assertEqual(len(case.nodes),9); self.assertEqual(len(case.loads),6)
        self.assertEqual(case.base_mva,100)
        nodes = {n.id:n for n in case.nodes}
        self.assertEqual(nodes['10.A'].vm,1.04); self.assertEqual(nodes['40.A'].vm,1.02)
        self.assertEqual(nodes['40.A'].p,.3); self.assertAlmostEqual(nodes['40.A'].qmin,-.3)
        self.assertAlmostEqual(nodes['40.A'].qmax,.7)
        self.assertEqual(nodes['700.A'].p,.04); self.assertEqual(nodes['700.A'].q,.02)
        self.assertEqual(nodes['700.A'].type,'PQ')
        self.assertNotIn('branch-3',{row[0] for row in case.stamps})
        self.assertAlmostEqual(sum(v.p for v in case.loads),3*.23)
        self.assertAlmostEqual(sum(v.q for v in case.loads),3*.07)
        expected = np.zeros((3,3),complex)
        y = 1/(.02+.1j); t = 1.03*np.exp(1j*np.deg2rad(5))
        expected[0,0] += (y+.02j)/1.03**2
        expected[0,1] -= y/np.conj(t); expected[1,0] -= y/t; expected[1,1] += y+.02j
        y = 1/(.015+.08j)
        expected[1,1] += y+.01j; expected[1,2] -= y
        expected[2,1] -= y; expected[2,2] += y+.01j
        expected[1,1] += .01-.02j; expected[2,2] += .02+.03j
        a = [net.index[f'{v}.A'] for v in (10,40,700)]
        b = [net.index[f'{v}.B'] for v in (10,40,700)]
        assert_allclose(net.y[a,:][:,a].toarray(),expected,atol=4e-15)
        assert_allclose(net.y[a,:][:,b].toarray(),0,atol=0)
        reference = independent_reference(data)
        result = solve(net,tolerance=1e-11)
        self.assertTrue(reference.converged,reference.reason); self.assertTrue(result.converged,result.reason)
        assert_allclose(result.voltage[a],reference.voltage,atol=2e-10)
        _,actual,_ = evaluate(net,result.voltage,result.types,jacobian=False)
        self.assertAlmostEqual(actual[net.index['700.A']].real,.04,delta=1e-10)
        self.assertAlmostEqual(actual[net.index['700.A']].imag,.02,delta=1e-10)
        assert_allclose(reference.generation_power[2],.04+.02j,atol=1e-10)

    def test_analytic_reference_is_independent_and_matches_exact_phasor(self):
        # This circuit's exact V9=.95-j.05 gives S9=-.5-j.45 through X=.1.
        with patch('nrlarge.solver.solve',side_effect=AssertionError('Reference used NRlarge solver')), \
             patch('nrlarge.model.evaluate',side_effect=AssertionError('Reference used NRlarge residual')), \
             patch('nrlarge.model.add_series',side_effect=AssertionError('Reference used NRlarge stamping')):
            result = independent_reference(analytic_case(),tolerance=1e-12)
        self.assertTrue(result.converged,result.reason)
        self.assertLess(result.residual,1e-12)
        assert_allclose(result.voltage,[1,.95-.05j],atol=2e-13)
        self.assertAlmostEqual(sum(result.generation_power).real,.5,delta=1e-12)
        self.assertAlmostEqual(sum(result.generation_power).imag,.5,delta=1e-12)

    def test_reference_q_limit_policy_matches_phase_engine(self):
        data = analytic_case(); data.bus[1,1:4] = [2,0,0]
        data.gen = np.vstack([data.gen,generator(9,40,qmax=10,qmin=-100,vg=1.02)])
        free = independent_reference(data,enforce_q_limits=False)
        limited = independent_reference(data)
        self.assertTrue(free.converged); self.assertTrue(limited.converged)
        self.assertEqual(free.types[1],'PV'); self.assertEqual(limited.types[1],'PQ')
        self.assertGreater(free.generator_q[1],.1)
        self.assertAlmostEqual(limited.generator_q[1],.1,delta=1e-10)
        net = Network(to_phase_case(data)); result = solve(net,tolerance=1e-11)
        self.assertTrue(result.converged,result.reason)
        self.assertEqual(result.types[net.index['9.A']],'PQ')
        assert_allclose(result.voltage[net.index['9.A']],limited.voltage[1],atol=2e-10)

    def test_controlled_sources_without_generator_and_conflicting_setpoints_reject(self):
        data = feature_case(); data.gen[0,7] = 0
        with self.assertRaises(ValueError): to_phase_case(data)
        data = feature_case(); data.gen[2,5] = 1.03
        with self.assertRaises(ValueError): to_phase_case(data)

    def test_all_pinned_standard_cases_polar_phase_agreement(self):
        cases = [('case14',14),('case_ieee30',30),('case57',57),('case118',118),('case300',300)]
        for name,count in cases:
            with self.subTest(case=name):
                data = load_matpower(SOURCES/(name+'.m'))
                self.assertEqual(len(data.bus),count)
                reference = independent_reference(data)
                self.assertTrue(reference.converged,reference.reason)
                self.assertLess(reference.residual,1e-10)
                net = Network(to_phase_case(data)); result = solve(net,tolerance=1e-10)
                self.assertTrue(result.converged,result.reason)
                for phase,degrees in zip('ABC',(0,-120,120)):
                    selected = [net.index[f'{identifier}.{phase}'] for identifier in reference.bus_ids]
                    expected = reference.voltage*np.exp(1j*np.deg2rad(degrees))
                    assert_allclose(result.voltage[selected],expected,rtol=0,atol=2e-7)
                    self.assertEqual(list(result.types[selected]),list(reference.types))
                original_pv = data.bus[:,1] == 2
                limited = original_pv & (reference.types == 'PQ')
                for k in np.flatnonzero(limited):
                    self.assertLess(min(abs(reference.generator_q[k]-reference.qmin[k]),
                                        abs(reference.generator_q[k]-reference.qmax[k])),1e-8)
                if name == 'case300': self.assertGreater(max(reference.bus_ids),300)


if __name__ == '__main__':
    unittest.main()
