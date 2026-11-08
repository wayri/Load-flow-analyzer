"""CLI, threaded Newton and bounded exports, using small task-owned scratch."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from nrlarge.__main__ import main
from nrlarge.examples import convert_balanced, synthetic, unbalanced_example
from nrlarge.linear import LocalBackend, RowShard
from nrlarge.model import Network, read_case, write_case
from nrlarge.reports import empty_output, svg, write_reports
from nrlarge.solver import solve


ROOT = Path(__file__).resolve().parents[2]


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run_cli(arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main([str(v) for v in arguments])
    return code, stdout.getvalue(), stderr.getvalue()


def write_legacy(directory, source="source", extra_bus_column=False):
    buses, branches = directory/"buses.csv", directory/"branches.csv"
    with buses.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow("id,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu".split(","))
        row = [source, "SLACK", 0, 0, 1, 0, "", ""]
        writer.writerow(row + (["unexpected"] if extra_bus_column else []))
        writer.writerow(["load", "PQ", -.5, -.45, 1, 0, "", ""])
    with branches.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow("id,from,to,r_pu,x_pu,b_pu,tap,shift_deg".split(","))
        writer.writerow(["line", source, "load", 0, .1, 0, 1, 0])
    return buses, branches


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/".task-check"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime-test-", dir=scratch)
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_sparse_direct_and_one_two_four_thread_gmres_newton_agree(self):
        net = Network(unbalanced_example())
        reference = solve(net, tolerance=1e-10)
        self.assertTrue(reference.converged, reference.reason)
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                backend = LocalBackend(threads=threads, method="gmres")
                try:
                    result = solve(net, backend, tolerance=1e-10)
                finally:
                    backend.close()
                self.assertTrue(result.converged, result.reason)
                assert_allclose(result.voltage, reference.voltage, rtol=0, atol=2e-10)
                self.assertLess(result.history[-1]["residual"], 1e-10)
                self.assertLess(result.history[-1]["power_error"], 1e-10)
                self.assertTrue(any(r.get("krylov_iterations", 0) > 0 for r in result.history))
                self.assertTrue(all(r.get("linear_residual", 0) < 1e-8 for r in result.history))

    def test_optional_gpu_full_newton_agrees_with_cpu_reference(self):
        from nrlarge.accelerators import AcceleratorUnavailable, create_accelerator
        net = Network(unbalanced_example())
        reference = solve(net, tolerance=1e-10)
        self.assertTrue(reference.converged, reference.reason)
        for kind in ("opencl", "opengl"):
            with self.subTest(accelerator=kind):
                # Only initial device/context unavailability may skip a backend.
                # Once established, every Newton/linear/numerical error must fail.
                try:
                    device = create_accelerator(kind, sparse.eye(2, format="csr"))
                except (AcceleratorUnavailable, OSError) as error:
                    self.skipTest(f"{kind} FP64 GPU unavailable: {error}")
                else:
                    device.close()
                backend = LocalBackend(threads=1, accelerator=kind)
                try:
                    result = solve(net, backend, tolerance=1e-10)
                finally:
                    backend.close()
                self.assertTrue(result.converged, result.reason)
                assert_allclose(result.voltage, reference.voltage, rtol=0, atol=2e-9)
                self.assertLess(result.history[-1]["residual"], 1e-10)
                self.assertLess(result.history[-1]["power_error"], 1e-10)

    def test_legacy_conversion_preserves_analytic_balanced_three_phase_solution(self):
        buses, branches = write_legacy(self.directory)
        case = convert_balanced(buses, branches, base_mva=60)
        net = Network(case)
        result = solve(net, tolerance=1e-11)
        self.assertTrue(result.converged, result.reason)
        self.assertEqual(len(case.nodes), 6)
        for phase, degrees in zip("ABC", (0, -120, 120)):
            index = net.index["load."+phase]
            expected = (.95-.05j)*np.exp(1j*np.deg2rad(degrees))
            assert_allclose(result.voltage[index], expected, rtol=0, atol=2e-11)
        summary = write_reports(net, result, self.directory/"balanced-report", {"mode": "test"})
        self.assertAlmostEqual(summary["p_loss_mw"], 0, delta=1e-10)
        self.assertAlmostEqual(summary["q_loss_mvar"], 3*.05*(60/3), delta=1e-9)
        loaded = read_case(self.directory/"balanced-report"/"inputs")
        self.assertEqual(loaded.base_mva, 60)
        self.assertEqual(loaded, case)

    def test_cli_conversion_rejects_empty_ids_and_wrong_record_width(self):
        for name, source, extra in (("empty-id", "", False), ("extra-field", "source", True)):
            with self.subTest(name=name):
                source_dir = self.directory/name; source_dir.mkdir()
                buses, branches = write_legacy(source_dir, source, extra)
                destination = source_dir/"converted"
                code, _, stderr = run_cli(["convert-balanced", "--buses", buses, "--branches", branches,
                                           "--out", destination])
                self.assertEqual(code, 1, stderr)
                self.assertIn("NRlarge error:", stderr)
                self.assertFalse(destination.exists(), "Invalid conversion must not export a partial case")

    def test_complete_report_power_balance_reload_diagram_and_focus_preserves_csv(self):
        case = unbalanced_example(); case.base_mva = 12.6
        case.nodes[-2].feeder = "F2"
        case.nodes[-1].customer = "neutral & customer"
        case.loads[-1].customer = 'shop, "quoted"'
        net = Network(case); result = solve(net, tolerance=1e-10)
        self.assertTrue(result.converged)
        full, focused = self.directory/"full", self.directory/"focused"
        summary = write_reports(net, result, full, {"mode": "test"}, max_plot_points=16)
        write_reports(net, result, focused, {"mode": "test"}, focus_feeder="F2", max_plot_points=16)
        self.assertFalse((full/"INCOMPLETE").exists())
        self.assertEqual(json.loads((full/"inputs"/"case.json").read_text())["base_mva"], 12.6)
        self.assertEqual(read_case(full/"inputs"), case)
        for name in ("nodes", "loads", "terminals", "feeders", "convergence"):
            filename = name+"-00000.csv"
            self.assertEqual((full/filename).read_bytes(), (focused/filename).read_bytes())
        nodes, loads, terminals = (csv_rows(full/(name+"-00000.csv")) for name in ("nodes", "loads", "terminals"))
        self.assertEqual(len(nodes), len(case.nodes)); self.assertEqual(len(loads), len(case.loads))
        self.assertTrue(all(row["converged"] == "1" for row in nodes+loads+terminals))
        generation = sum(float(row["p_generation_mw"]) for row in nodes)
        demand = sum(float(row["p_mw"]) for row in loads)
        terminal_loss = sum(float(row["p_pu"]) for row in terminals)*(case.base_mva/3)
        self.assertAlmostEqual(generation-demand, summary["p_loss_mw"], delta=1e-9)
        self.assertAlmostEqual(terminal_loss, summary["p_loss_mw"], delta=1e-9)
        self.assertGreater(summary["p_loss_mw"], 0)
        diagram = (full/"network.txt").read_text()
        for node in case.nodes: self.assertIn("["+node.id+"]", diagram)
        for load in case.loads:
            self.assertIn("<LOAD "+load.id, diagram)
            self.assertIn("["+load.source+"]", diagram)
            self.assertIn("["+load.target+"]", diagram)
        focused_svg = (focused/"voltage.svg").read_text()
        self.assertIn("1 sampled of 1", focused_svg)
        self.assertIn("service.C", focused_svg)
        self.assertNotIn("service.A", focused_svg)

    def test_svg_sampling_is_bounded_and_input_arrays_are_unchanged(self):
        values = np.arange(12000, dtype=float)
        original = values.copy()
        path = self.directory/"large-values.svg"
        svg(path, "A < B & C", values, limit=4096)
        tree = ElementTree.parse(path)
        polyline = tree.find(".//{http://www.w3.org/2000/svg}polyline")
        self.assertEqual(len(polyline.attrib["points"].split()), 4096)
        self.assertIn("4096 sampled of 12000", path.read_text())
        self.assertIn("A &lt; B &amp; C", path.read_text())
        assert_allclose(values, original, atol=0)

    def test_nonconverged_exports_and_refusal_to_overwrite(self):
        net = Network(unbalanced_example())
        result = solve(net, tolerance=1e-14, max_iterations=1)
        self.assertFalse(result.converged)
        out = self.directory/"last-iterate"
        summary = write_reports(net, result, out, {"mode": "test"})
        self.assertFalse(summary["converged"])
        for name in ("nodes", "loads", "terminals"):
            self.assertTrue(all(row["converged"] == "0" for row in csv_rows(out/(name+"-00000.csv"))))
        self.assertIn("NOT CONVERGED - LAST ITERATE ONLY", (out/"network.txt").read_text())
        self.assertIn("NOT CONVERGED", (out/"voltage.svg").read_text())
        receipt = (out/"summary.json").read_bytes()
        with self.assertRaises(ValueError): write_reports(net, result, out, {})
        self.assertEqual((out/"summary.json").read_bytes(), receipt)
        sentinel = self.directory/"sentinel.txt"; sentinel.write_text("keep")
        with self.assertRaises(ValueError): empty_output(sentinel)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_memory_and_output_preflights_reject_before_export(self):
        with self.assertRaises(MemoryError): synthetic(1000000, memory_gib=1e-5)
        case = unbalanced_example(); inputs = self.directory/"input"; write_case(case, inputs)
        with self.assertRaises(MemoryError): read_case(inputs, memory_gib=1e-8)
        net = Network(case); result = solve(net)
        destination = self.directory/"too-small"
        with self.assertRaises(ValueError): write_reports(net, result, destination, {}, max_output_mib=.001)
        self.assertFalse(destination.exists())
        for points in (1, 4097):
            with self.assertRaises(ValueError): write_reports(net, result, destination, {}, max_plot_points=points)
            self.assertFalse(destination.exists())
        code, _, stderr = run_cli(["solve", "--case", inputs, "--out", destination, "--memory-gib", "0.00000001"])
        self.assertEqual(code, 1); self.assertIn("RAM budget", stderr); self.assertFalse(destination.exists())

    def test_cli_example_solve_nonconvergence_and_benchmark(self):
        inputs = self.directory/"editable"
        code, _, stderr = run_cli(["example", "--out", inputs])
        self.assertEqual(code, 0, stderr)
        out = self.directory/"solved"
        code, stdout, stderr = run_cli(["solve", "--case", inputs, "--out", out, "--threads", "2"])
        self.assertEqual(code, 0, stderr)
        summary = json.loads(stdout)
        self.assertTrue(summary["converged"])
        self.assertEqual(summary["execution"]["threads"], 2)
        receipt = (out/"summary.json").read_bytes()
        code, _, stderr = run_cli(["example", "--out", out])
        self.assertEqual(code, 1); self.assertIn("new or empty", stderr)
        self.assertEqual((out/"summary.json").read_bytes(), receipt)
        code, stdout, stderr = run_cli(["solve", "--case", inputs, "--out", self.directory/"failed",
                                      "--max-iterations", "1", "--tolerance", "1e-14", "--linear", "direct"])
        self.assertEqual(code, 2, stderr); self.assertFalse(json.loads(stdout)["converged"])
        code, stdout, stderr = run_cli(["benchmark", "--customers", "7", "--threads", "2"])
        self.assertEqual(code, 0, stderr)
        summary = json.loads(stdout)
        self.assertIn("SYNTHETIC", summary["dataset"])
        self.assertEqual(summary["customers"], 7); self.assertTrue(summary["converged"])

    def test_interleaved_element_stamps_are_rejected_before_streamed_reports(self):
        case = unbalanced_example()
        # Split one element around another to exercise the documented streaming constraint.
        case.stamps.insert(1, ("other-element", case.nodes[0].id, case.nodes[0].id, .01j))
        with self.assertRaises(ValueError): Network(case)

    def test_worker_uses_multiple_cpu_threads_for_uneven_rectangular_rows(self):
        matrix = sparse.csr_matrix([[2,0,1,0,0,0,3],[0,1,0,0,2,0,0],
                                    [0,0,0,0,0,0,0],[1,2,3,4,5,6,7],[0,0,0,1,0,0,0]],dtype=float)
        worker = RowShard(matrix,sparse.eye(5,format='csc'),threads=3)
        vector = np.arange(7,dtype=float)
        try:
            self.assertEqual(len(worker.chunks),3)
            for _ in range(3):
                assert_allclose(worker.matvec(vector),matrix @ vector,rtol=1e-14,atol=1e-14)
            assert_allclose(worker.precondition(np.arange(5,dtype=float)),np.arange(5))
        finally:
            worker.close(); worker.close()
        self.assertIsNone(worker.pool)


if __name__ == "__main__":
    unittest.main()
