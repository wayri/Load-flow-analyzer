"""Independent phase-domain numerical contracts; no optional GPU dependency."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from nrlarge.model import (
    Case, Load, Network, Node, add_series, current_and_derivatives, evaluate,
    read_case, write_case,
)
from nrlarge.solver import solve


def phasor(magnitude, degrees):
    return magnitude * np.exp(1j * np.deg2rad(degrees))


def mixed_case():
    """Unequal phasors, mutual coupling, complex taps, ZIP wye/delta and PV."""
    nodes = [Node("SA", phase="A", type="SLACK", vm=1.03, va=2),
             Node("SB", phase="B", type="SLACK", vm=.99, va=-121),
             Node("SC", phase="C", type="SLACK", vm=1.01, va=119),
             Node("LA", phase="A", vm=.97, va=-1),
             Node("LB", phase="B", p=.03, q=.02, vm=.94, va=-123),
             Node("LC", phase="C", type="PV", p=.12, vm=1.01, va=117)]
    impedance = np.array([[.02+.12j, .004+.018j, .003+.013j],
                          [.004+.018j, .026+.135j, .005+.016j],
                          [.003+.013j, .005+.016j, .018+.11j]])
    stamps = []
    add_series(stamps, "coupled", ["SA", "SB", "SC"], ["LA", "LB", "LC"],
               impedance, charging=np.diag([.002j, .003j, .001j]),
               taps=[phasor(1.02, 2), phasor(.98, -1), phasor(1.01, .5)])
    loads = [Load("wye-A", "LA", p=.22, q=.08, z=.2, i=.3, power=.5),
             Load("wye-B", "LB", p=.14, q=.05, z=0, i=.4, power=.6),
             Load("delta-AB", "LA", "LB", .09, .04, .25, .15, .6, math.sqrt(3)),
             Load("delta-BC", "LB", "LC", .07, .02, .3, .7, 0, math.sqrt(3))]
    return Case(nodes, stamps, loads, base_mva=12.0)


def neutral_case():
    """Analytic two-wire circuit with nonzero neutral displacement."""
    current = .5-.2j
    za, zn = .01+.08j, .02+.04j
    va, vn = 1-za*current, zn*current
    consumption = (va-vn)*np.conj(current)
    nodes = [Node("SA", phase="A", type="SLACK"),
             Node("SN", phase="N", type="SLACK", vm=0),
             Node("LA", phase="A"),
             Node("LN", phase="N", vm=.02, va=45)]
    stamps = []
    add_series(stamps, "phase-neutral", ["SA", "SN"], ["LA", "LN"], np.diag([za, zn]))
    load = Load("single-phase", "LA", "LN", consumption.real, consumption.imag)
    return Case(nodes, stamps, [load]), np.array([1, 0, va, vn]), current


def delta_case():
    current = .2-.1j
    source_a, source_b = 1+0j, phasor(1, -120)
    za, zb = .02+.1j, .03+.08j
    va, vb = source_a-za*current, source_b+zb*current
    consumption = (va-vb)*np.conj(current)
    nodes = [Node("SA", phase="A", type="SLACK"),
             Node("SB", phase="B", type="SLACK", va=-120),
             Node("LA", phase="A"), Node("LB", phase="B", va=-120)]
    stamps = []
    add_series(stamps, "two-phases", ["SA", "SB"], ["LA", "LB"], np.diag([za, zb]))
    load = Load("delta", "LA", "LB", consumption.real, consumption.imag, nominal=math.sqrt(3))
    return Case(nodes, stamps, [load]), np.array([source_a, source_b, va, vb]), current


class ModelTests(unittest.TestCase):
    def check_jacobian(self, net, voltage, types=None, q=None):
        f, jacobian, _, _ = evaluate(net, voltage, types, q)
        self.assertTrue(sparse.isspmatrix_csr(jacobian))
        finite_difference = np.empty(jacobian.shape)
        h = 1e-6
        for k in range(2*len(net.free)):
            plus, minus = voltage.copy(), voltage.copy()
            increment = h if k % 2 == 0 else 1j*h
            plus[net.free[k//2]] += increment
            minus[net.free[k//2]] -= increment
            fp = evaluate(net, plus, types, q, jacobian=False)[0]
            fm = evaluate(net, minus, types, q, jacobian=False)[0]
            finite_difference[:, k] = (fp-fm)/(2*h)
        self.assertEqual(f.shape, (2*len(net.free),))
        assert_allclose(jacobian.toarray(), finite_difference, rtol=2e-7, atol=2e-8)

    def test_unequal_wye_delta_zip_mutual_taps_pv_jacobian(self):
        net = Network(mixed_case())
        voltage = net.initial.copy()
        voltage[3:] *= [phasor(.99, -.7), phasor(1.02, .8), phasor(.98, 1.3)]
        self.check_jacobian(net, voltage)
        types, q = net.types.copy(), net.q.copy()
        types[-1] = "PQ"; q[-1] = .17
        self.check_jacobian(net, voltage, types, q)

    def test_explicit_neutral_jacobian_and_exact_circuit(self):
        case, voltage, expected_current = neutral_case()
        net = Network(case)
        residual, _, actual, currents = evaluate(net, voltage)
        assert_allclose(residual, 0, atol=1e-13)
        assert_allclose(currents, [expected_current], atol=1e-13)
        assert_allclose(actual[net.free], 0, atol=1e-13)
        self.assertGreater(abs(voltage[-1]), .02)
        self.check_jacobian(net, voltage)

    def test_delta_exact_circuit_and_equal_opposite_terminal_currents(self):
        case, voltage, expected_current = delta_case()
        net = Network(case)
        residual, _, actual, currents = evaluate(net, voltage)
        assert_allclose(residual, 0, atol=2e-14)
        assert_allclose(currents, [expected_current], atol=2e-14)
        demand = net.a.T @ currents
        assert_allclose(demand, [0, 0, expected_current, -expected_current], atol=2e-14)
        assert_allclose(actual[net.free], 0, atol=2e-14)
        self.check_jacobian(net, voltage)

    def test_zip_power_definition_and_total_complex_power_balance(self):
        net = Network(mixed_case())
        _, _, actual, load_currents = evaluate(net, net.initial)
        terminals = net.a @ net.initial
        ratios = np.abs(terminals)/net.nominal
        expected_power = net.power*(net.z*ratios**2+net.i*ratios+net.cp)
        assert_allclose(terminals*np.conj(load_currents), expected_power, atol=2e-15)
        passive_absorption = np.sum(net.initial*np.conj(net.y @ net.initial))
        assert_allclose(np.sum(actual), passive_absorption+np.sum(expected_power), atol=2e-14)
        # Individual terminal allocation must conserve complex power for every delta load.
        for k, current in enumerate(load_currents):
            demand = net.a.getrow(k).toarray().ravel()*current
            assert_allclose(np.sum(net.initial*np.conj(demand)), expected_power[k], atol=2e-15)

    def test_complex_tap_impedance_stamp_from_terminal_equations(self):
        case = mixed_case()
        net = Network(case)
        z = np.array([[.02+.12j, .004+.018j, .003+.013j],
                      [.004+.018j, .026+.135j, .005+.016j],
                      [.003+.013j, .005+.016j, .018+.11j]])
        taps = np.array([phasor(1.02, 2), phasor(.98, -1), phasor(1.01, .5)])
        vf, vt = net.initial[:3], net.initial[3:]
        current = np.linalg.solve(z, vf/taps-vt)
        shunt = np.array([.001j, .0015j, .0005j])
        expected = np.r_[(current+shunt*vf/taps)/np.conj(taps), -current+shunt*vt]
        assert_allclose(net.y @ net.initial, expected, atol=4e-15)

    def test_csv_reload_preserves_electrical_and_partition_metadata(self):
        case = mixed_case()
        case.nodes[3].id = 'LA, "quoted"'
        case.nodes[3].region = "north & east"
        case.nodes[3].feeder = "F-1"
        case.nodes[3].customer = "customer 27"
        case.stamps = [(e, case.nodes[3].id if a == "LA" else a,
                        case.nodes[3].id if b == "LA" else b, y) for e, a, b, y in case.stamps]
        for load in case.loads:
            if load.source == "LA": load.source = case.nodes[3].id
            if load.target == "LA": load.target = case.nodes[3].id
        case.loads[0].customer = "27, main"
        # Keep this test's few-kilobyte scratch inside its writable checkout.
        with tempfile.TemporaryDirectory(prefix=".model-test-", dir=Path(__file__).parent) as directory:
            write_case(case, directory)
            loaded = read_case(directory, base_mva=case.base_mva)
        self.assertEqual(loaded, case)
        before, after = Network(case), Network(loaded)
        assert_allclose(before.y.toarray(), after.y.toarray(), atol=0)
        assert_allclose(evaluate(before, before.initial)[0], evaluate(after, after.initial)[0], atol=0)

    def test_validation_and_preflight_memory_cap(self):
        def rejected(change, error=ValueError):
            case = mixed_case(); change(case)
            with self.assertRaises(error): Network(case)
        rejected(lambda c: setattr(c.nodes[1], "id", c.nodes[0].id))
        rejected(lambda c: setattr(c.nodes[0], "id", "GROUND"))
        rejected(lambda c: setattr(c.nodes[3], "phase", "D"))
        rejected(lambda c: setattr(c.nodes[3], "p", math.nan))
        rejected(lambda c: (setattr(c.nodes[5], "qmin", 2), setattr(c.nodes[5], "qmax", 1)))
        rejected(lambda c: setattr(c.nodes[5], "qmax", -math.inf))
        rejected(lambda c: setattr(c.nodes[5], "vm", 0))
        rejected(lambda c: setattr(c.loads[0], "source", "unknown"))
        rejected(lambda c: setattr(c.loads[0], "target", c.loads[0].source))
        rejected(lambda c: setattr(c.loads[0], "id", c.loads[1].id))
        rejected(lambda c: setattr(c.loads[0], "power", .9))
        rejected(lambda c: setattr(c.loads[0], "nominal", 0))
        rejected(lambda c: setattr(c.loads[0], "z", -.2))
        rejected(lambda c: setattr(c, "base_mva", 0))
        case = mixed_case()
        net = Network(case)
        with self.assertRaises(MemoryError):
            Network(case, memory_gib=net.estimated_bytes/2**30*.99)
        Network(case, memory_gib=net.estimated_bytes/2**30*1.01)
        for memory in (0, math.nan, -1):
            with self.assertRaises(ValueError): Network(case, memory_gib=memory)
        with self.assertRaises(ValueError): Network(case, restart=1)
        unreferenced = Case([Node("alone")], [])
        with self.assertRaises(ValueError): Network(unreferenced)

    def test_terminal_voltage_collapse_and_zero_power_device(self):
        with self.assertRaises(ValueError):
            current_and_derivatives(np.array([0j]), np.array([.1+.1j]), 0, 0, 1, 1)
        current, de, df = current_and_derivatives(np.array([0j]), np.array([.1+.1j]), 1, 0, 0, 1)
        assert_allclose(current, [0j], atol=0)
        assert_allclose(de, [.1-.1j], atol=0)
        assert_allclose(df, [.1+.1j], atol=0)
        current, _, _ = current_and_derivatives(np.array([0j]), np.array([0j]), 0, 0, 1, 1)
        assert_allclose(current, [0j], atol=0)

    def test_load_connectivity_cannot_cancel_admittance_connectivity(self):
        # A ZIP impedance device in parallel with an existing series element
        # must add connectivity, never remove it through signed incidence terms.
        stamps = []
        add_series(stamps, "line", ["S"], ["L"], [[.1j]])
        case = Case([Node("S", type="SLACK"), Node("L")], stamps,
                    [Load("parallel", "S", "L", .01, .001, z=1, power=0)])
        net = Network(case)
        assert_allclose(evaluate(net, net.initial, jacobian=False)[0], 0, atol=0)


class SolverTests(unittest.TestCase):
    def test_analytic_delta_and_explicit_neutral_solutions(self):
        for fixture in (delta_case, neutral_case):
            with self.subTest(fixture=fixture.__name__):
                case, expected_voltage, expected_current = fixture()
                net = Network(case)
                initial = net.initial.copy()
                result = solve(net, tolerance=1e-10, max_iterations=100)
                self.assertTrue(result.converged, result.reason)
                assert_allclose(result.voltage, expected_voltage, atol=2e-10, rtol=0)
                residual, actual, currents = evaluate(net, result.voltage, result.types, jacobian=False)
                assert_allclose(residual, 0, atol=1e-10)
                assert_allclose(currents, [expected_current], atol=1e-10)
                assert_allclose(actual[net.free], 0, atol=1e-10)
                assert_allclose(net.initial, initial, atol=0)
                self.assertLess(result.history[-1]['residual'], 1e-10)
                self.assertLess(result.history[-1]['power_error'], 1e-10)

    def test_unbalanced_mutual_zip_pv_solution_and_physical_balance(self):
        net = Network(mixed_case())
        result = solve(net, tolerance=1e-10, max_iterations=100)
        self.assertTrue(result.converged, result.reason)
        _, actual, currents = evaluate(net, result.voltage, result.types, jacobian=False)
        assert_allclose(actual[3:5], [0, .03+.02j], atol=1e-10)
        self.assertAlmostEqual(abs(result.voltage[5]), 1.01, delta=1e-10)
        self.assertAlmostEqual(actual[5].real, .12, delta=1e-10)
        # Check each phase source against the physical passive/load currents.
        passive_current = net.y @ result.voltage
        load_current = np.zeros(len(net.case.nodes), dtype=complex)
        load_power = 0j
        for k, device in enumerate(net.case.loads):
            a = net.index[device.source]
            b = None if device.target == "GROUND" else net.index[device.target]
            terminal = result.voltage[a] - (0 if b is None else result.voltage[b])
            ratio = abs(terminal)/device.nominal
            consumption = complex(device.p, device.q)*(device.z*ratio**2+device.i*ratio+device.power)
            expected_current = np.conj(consumption/terminal)
            assert_allclose(currents[k], expected_current, atol=1e-13)
            load_current[a] += expected_current
            if b is not None: load_current[b] -= expected_current
            load_power += consumption
        assert_allclose(actual, result.voltage*np.conj(passive_current+load_current), atol=1e-13)
        passive_power = np.sum(result.voltage*np.conj(passive_current))
        assert_allclose(np.sum(actual)-load_power, passive_power, atol=1e-13)
        self.assertGreater(passive_power.real, 0)

    def test_pv_feasible_and_upper_lower_reactive_limits(self):
        stamps = []
        add_series(stamps, "line", ["S"], ["G"], [[.1j]])
        for minimum, maximum, expected_type, expected_q in (
            (.21, .22, "PV", None), (-1, .1, "PQ", .1), (.3, 1, "PQ", .3)
        ):
            with self.subTest(minimum=minimum, maximum=maximum):
                net = Network(Case([Node("S", type="SLACK"),
                                    Node("G", type="PV", p=.4, vm=1.02, qmin=minimum, qmax=maximum)], stamps))
                result = solve(net, tolerance=1e-10)
                self.assertTrue(result.converged, result.reason)
                self.assertEqual(result.types[1], expected_type)
                _, actual, _ = evaluate(net, result.voltage, jacobian=False)
                self.assertAlmostEqual(actual[1].real, .4, delta=1e-10)
                if expected_q is None:
                    self.assertAlmostEqual(abs(result.voltage[1]), 1.02, delta=1e-10)
                    self.assertLess(actual[1].imag, maximum)
                    self.assertGreater(actual[1].imag, minimum)
                else:
                    self.assertAlmostEqual(actual[1].imag, expected_q, delta=1e-10)
                self.assertEqual(net.types[1], "PV", "solve must not mutate reusable network")

    def test_iteration_exhaustion_invalid_settings_and_all_reference_case(self):
        case, _, _ = neutral_case()
        net = Network(case)
        result = solve(net, tolerance=1e-12, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "iteration_limit")
        self.assertGreater(max(result.history[-1]['residual'], result.history[-1]['power_error']), 1e-12)
        for tolerance in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError): solve(net, tolerance=tolerance)
        with self.assertRaises(ValueError): solve(net, max_iterations=0)
        reference = Network(Case([Node("S", type="SLACK")], []))
        solved = solve(reference)
        self.assertTrue(solved.converged)
        self.assertEqual(len(solved.history), 1)
        self.assertEqual(solved.history[-1]['residual'], 0)


if __name__ == "__main__":
    unittest.main()
