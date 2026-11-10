# Benchmark validation receipt

Verified on 10 November 2026 using the environment recorded in
[the run manifest](results/manifest.json).

- All eight CTest groups passed in 30.02 seconds: two existing C++ groups and six
  Python groups, including 64 Python tests with no skips.
- All five IEEE baselines and both derived balanced scenarios converge and meet
  the 2e-7 pu complex-voltage comparison tolerance against the independent polar
  implementation. The largest baseline error is 3.31e-13 pu.
- All six normal LV snapshots converge. The extreme neutral-impedance case
  returns its expected nonconverged status and labelled final iterate.
- CPU sparse LU, one/four-thread GMRES, OpenCL and OpenGL IEEE 14 runs pass the
  voltage agreement check. Actual per-run timings and device details are in
  [the backend table](results/backend-comparison.csv) and
  [hardware receipt](results/hardware.json).
- All 421 published output files pass exact-byte hash, status, reference,
  PNG/SVG payload and local-link verification; 73 PNG previews are included.
- Representative IEEE 300 topology, EV and rooftop voltage profiles, collection
  comparison and failed-case power-balance previews were visually inspected.
- Pinned raw source files retain their upstream bytes. Hand-authored source is
  whitespace-checked; vendor data and generated outputs use byte-hash checks.

The case importer includes shunts, charging, complex transformer taps, status
flags and aggregate online generator controls. Analytic fixtures verify the
per-phase power-base conversion and independent reference equations.

A PV-only regression exposed integer truncation of fractional Q limits. The
canonical model arrays now use floating-point storage, and a standalone test
checks the fractional clamp. Scenario review also corrected rooftop generation
to return through the actual neutral and added explicit expected-outcome gates.

Recheck the collection:

```sh
python -m benchmarks.verify
```

[Gallery](results/README.md) · [Acknowledgements, disclaimer and waiver](../ACKNOWLEDGEMENTS.md)
