# NRlarge verification evidence

Verified on 9 November 2026. This receipt records bounded evidence, not national
capacity or a speedup guarantee.

## Environment

- Windows, Python 3.14; NumPy 2.4.2, SciPy 1.17.0.
- Intel Arc 140T GPU (16 GB); Intel OpenCL Graphics FP64.
- Actual OpenGL renderer Intel Arc 140T; OpenGL 4.3.0 Build 32.0.101.8991.
- Existing native OpenCL loader and PyOpenGL; Windows OpenGL used a hidden WGL
  context. No package, driver or SDK was installed for this task.
- CPU reference/direct sparse LU, 1/2/4-thread GMRES and both GPU products use
  float64; final physical and true linear residuals are checked on CPU.

## Numerical and runtime checks

Local verification passed all six CTest entries (two existing C++ checks and
four Python suites), including **52 Python tests with zero skips**.

The suites cover:

- Analytic rectangular Jacobian versus central finite differences with unequal
  phases, wye/delta ZIP loads, mutual impedance, complex taps, PV and converted PQ.
- Exact delta and explicit-neutral circuits, including neutral displacement.
- Physical phase currents/powers, ZIP powers, transformer terminal equations,
  PV setpoints, both Q-limit directions, nonconvergence and invalid data.
- Original balanced two-bus conversion: each phase reproduces 0.95-j0.05 pu and
  the expected reactive loss; balanced base conversion is checked.
- Sparse direct versus 1/2/4-thread solves; whole-Newton OpenCL/OpenGL versus CPU
  voltage agreement within 2e-9 and residual/power error below 1e-10.
- GPU rectangular/int64 CSR products, duplicate/empty rows, repeated products,
  invalid/non-finite data, overflow detection, context restoration, thread use and
  idempotent cleanup. GPU selection does not fall back silently.
- Two real spawned local worker processes; distributed GMRES/reference agreement;
  verified TLS and authentication; exact chunked CSR coverage, stale sequence/job/
  epoch rejection, incomplete commit, lease expiry and staged release.
- Corrupt/wrong worker results, malformed frames, unavailable workers, redirects,
  wrong certificates/tokens, bounded TLS handshakes/request headers and recovery.
- CSV reload with base/metadata, full conductor/customer/terminal reports and power
  balances, ASCII coverage, bounded SVG sampling, feeder selection, output/memory
  preflights, failure flags and overwrite protection.

Commands (existing dependencies required):

```sh
python -m unittest discover -s tests/nrlarge -v
python -m unittest discover -s tests -p test_nrlarge_gpu.py -v
```

The final Python recheck passed all four suites. A separate launch without the
MinGW runtime directory on PATH initially failed the two existing C++ executables
at DLL loading; restoring the compiler runtime path made both pass. CTest now
prepends that directory automatically for MinGW with CMake 3.22 or newer; older
CMake/MinGW environments must put the compiler runtime directory on PATH.

CMake also registers these suites when Python/NumPy/SciPy are available. Use
`-DNRLARGE_REQUIRE_PYTHON_TESTS=ON` to fail configuration rather than omit them.
Device/context and optional cryptography test-certificate prerequisites report
explicit skips where absent. Hosted C++ CI does not install Python dependencies;
a passing C++ build alone is not evidence that the Python suites ran.

## Synthetic capacity receipt

Command:

```sh
python -m nrlarge benchmark --customers 100000 --threads 4 --memory-gib 3 --tolerance 1e-10
```

Measured receipt from the current RCM-ordered engine:

```json
{
  "dataset": "SYNTHETIC, not national data",
  "customers": 100000,
  "nodes": 203132,
  "y_nnz": 1237552,
  "estimated_peak_gib": 1.991049274802208,
  "seconds": 20.57463149999967,
  "iterations": 1,
  "converged": true,
  "reason": "converged",
  "residual": 1.233456012600788e-12,
  "power_error": 1.2461034976739295e-12,
  "execution": {
    "mode": "local",
    "threads": 4,
    "linear": "gmres",
    "accelerator": "cpu",
    "ordering": "rcm"
  },
  "setup_seconds": 20.92515039999853,
  "total_seconds": 41.5136222000001
}
```

The 1.991 GiB value is the **preflight estimate**, not measured peak RSS. Available
physical RAM was checked before the run (about 12 GiB). No full 100,000-customer
CSV/report dataset was written. The model has short, lightly loaded service
circuits and many feeders attached to ideal fixed-voltage sources; it converges
in one Newton update. It is a favorable synthetic case, not a realistic national
mesh, severe operating point, scalability curve or evidence for ten million
customers. Timings vary with contention, ordering, drivers and topology.

## Coverage not established

- Physical multi-host operation across 10 or 20 Windows/Linux computers: not run;
  no host endpoints/certificates were supplied. Transport was exercised using
  actual independent local processes and verified TLS.
- Real US-state or Indian regional-grid/customer data: not supplied or validated.
- National-scale conditioning, difficult loadability/contingency cases, true
  distributed model storage, optimal partitions and production availability.
- Linux GPU-driver coverage, GPU speedup, multi-GPU scaling, and end-to-end strong/
  weak scaling on a physical cluster.
- Automatic transformer vector-group/neutral/grounding validation, aggregate
  synchronous-generator excitation models, regulator control or dynamic studies.

These gaps must remain visible before making operational or national-capacity
claims. The branch provides an executable, measured foundation and deployable
numeric worker protocol, with coordinator-memory and transport limits described
in the [guide](NRlarge.md).
