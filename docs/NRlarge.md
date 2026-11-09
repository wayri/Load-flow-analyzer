# NRlarge: sparse unbalanced load flow

NRlarge is a phase-domain **steady-state** Newton-Raphson engine. It adds individual
customer connections, explicit neutral conductors, coupled phase impedances, ZIP
loads, local CPU parallelism, numerical workers on up to 20 configured computers,
and optional double-precision OpenCL/OpenGL sparse products. It runs on Windows
and Linux with Python, NumPy and SciPy. The original balanced C++ application and
its CSV/diagram/plot capabilities remain unchanged on this branch.

The engine uses a coordinator-based distributed solve with no fixed bus-count
cap. Capacity depends on coordinator RAM, sparsity, conditioning, worker RAM,
GPU limits, network traffic and required accuracy. Current verification covers a
100,000-customer synthetic case and independent local worker processes. Real
regional-grid models and physical multi-host clusters are listed in the
[remaining validation coverage](NRlarge-verification.md#coverage-not-established).

See [Acknowledgements, disclaimer and waiver](../ACKNOWLEDGEMENTS.md).

## Start locally

Use an existing Python 3.10+ environment with the dependencies in
`requirements-nrlarge.txt`. The verified local environment used Python 3.14,
NumPy 2.4.2 and SciPy 1.17.0. Before any new environment or substantial installation,
retain the required 100 GiB free on each affected drive; no packages were installed
for this implementation.

```sh
python -m nrlarge --help
python -m nrlarge example --out my_case
python -m nrlarge solve --case my_case --out my_results --threads 4
python -m nrlarge solve --case examples/unbalanced --out example_results --threads 4
```

Outputs require a new/empty directory. Exit 0 means convergence; exit 1 means invalid
data, I/O, device, transport or linear-solver failure; exit 2 means nonlinear
nonconvergence. Every result CSV and plot marks a failed solve. Reports interrupted
by an export error retain an `INCOMPLETE` marker rather than a success receipt.

The generated case is editable and includes an unequal three-phase four-wire
service, neutral displacement, mutual impedance and a phase-to-phase delta load.
Its per-unit demand is intentionally a visible numerical example, not a household
consumption forecast.

To reuse the original balanced CSV schemas:

```sh
python -m nrlarge convert-balanced --buses examples/three_bus/buses.csv --branches examples/three_bus/branches.csv --base-mva 100 --out expanded_case
python -m nrlarge solve --case expanded_case --out expanded_results --threads 4
```

Each original bus becomes three phase conductors at angles 0/-120/+120 degrees plus
the original angle. Each phase retains the original per-unit P/Q on a **per-phase
base of total three-phase MVA /3**, so the sum of physical MW agrees with the
original balanced model. Line charging, taps, phase shifts, PV setpoints and Q
limits are preserved. The converter does not introduce phase imbalance or invent
customer data.

## Electrical data contract

A case directory contains `case.json`, `nodes.csv`, `stamps.csv` and optional
`loads.csv`. CSV is UTF-8 with optional BOM; exact headers/order are required.
Quoted fields and CRLF are supported. Fields are limited to 512 characters and
IDs are case-sensitive. IDs do not have to be numeric or sequential.

```json
{"schema":1,"base_mva":100.0}
```

`base_mva` is the **total three-phase** base; each conductor's power base is one
third of it. All supplied impedances/admittances/voltages must be converted to
consistent per-unit quantities before import. Different voltage levels need
consistent local voltage bases and corresponding transformer stamps. No kV or
MVA base is inferred from a bus name.

`nodes.csv`:

```csv
id,bus,phase,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu,region,feeder,customer
source.A,source,A,SLACK,0,0,1,0,,,North,F1,
customer.A,customer,A,PQ,0,0,1,0,,,North,F1,C001
```

| Fields | Meaning |
| --- | --- |
| `id`, `bus`, `phase` | Unique conductor ID, physical bus/group, A/B/C/N/S1/S2 phase label |
| `type` | SLACK fixed complex voltage, PQ specified injection, PV per-conductor P and magnitude control |
| `p_pu`, `q_pu` | Positive generation/injection, negative fixed net demand |
| `vm_pu`, `va_deg` | Initial voltage magnitude and angle; fixed for SLACK, magnitude setpoint for PV |
| `qmin_pu`, `qmax_pu` | PV per-conductor source reactive-injection bounds; blanks are unbounded |
| `region`, `feeder`, `customer` | Optional identifiers carried through exports and feeder aggregation |

Use node P/Q for specified injections and `loads.csv` for explicit customer
consumption; do not enter the same demand twice. With separate loads at a PV
node, Q limits constrain the computed generator injection including that demand.
Per-phase PV controls are **independent controls**, not a total three-phase
synchronous-generator excitation model. SLACK P/Q are not controlled and its Q
limits are not enforced. Each structural electrical component needs a SLACK
reference; a zero-volt source-neutral reference is permitted with zero P/Q.
Neutral and split-phase labels alone do not create grounding or transformer
connections: their admittances and source phasors must actually be supplied.

`loads.csv`:

```csv
id,from,to,p_pu,q_pu,z_fraction,i_fraction,p_fraction,nominal_vm_pu,customer,feeder,region
L1,customer.A,GROUND,0.00009,0.000027,0.2,0.3,0.5,1,C001,F1,North
```

Positive load P/Q means consumption. A wye load connects phase to `GROUND` or an
explicit neutral conductor. A delta load connects two phase IDs; set its nominal
terminal magnitude consistently (sqrt(3) for a balanced line-to-line voltage on a
line-to-neutral voltage base). The Z/I/P fractions are nonnegative and sum to 1.
Their complex nominal power is scaled by `(V/Vnom)^2`, `abs(V)/Vnom`, and 1,
respectively. Zero-power devices are allowed at zero voltage; nonzero I/P devices
at collapsed terminal voltage cause failure. Negative load power can describe an
injection, but no dynamic inverter or control behavior is implied.

`stamps.csv`:

```csv
element,row,col,g_pu,b_pu
line1,source.A,source.A,0,-10
line1,source.A,customer.A,0,10
line1,customer.A,source.A,0,10
line1,customer.A,customer.A,0,-10
```

Each row adds `g+j*b` to a global conductor admittance entry. Parallel entries
are summed. **All rows for one element must be contiguous** so terminal-power and
ASCII reports can stream one element at a time. General stamps support mutual
coupling, shunts, explicit neutrals and transformer terminal relationships; they
are not assumed symmetric because phase-shifting transformers need not be.
Supplying a stamp is not validation of a winding/vector-group model.

The Python `add_series()` helper builds a coupled multi-conductor series element
from a small impedance matrix, optional total charging matrix and complex
from-side tap ratios. `examples.py` shows its four-wire use. Ideal zero-impedance
connections should be merged or modeled appropriately, not inverted as a line.
General winding, grounding, split-phase and device primitives require independent
reference validation before engineering use.

## Sparse Newton method and multicore CPU execution

The engine solves rectangular real/imaginary conductor voltages. PQ/neutral rows
use nodal current balance; PV rows use active-power balance and squared voltage
magnitude. ZIP and delta cross-phase derivatives are analytic. Both the residual
and an explicit PQ complex-power error must satisfy `--tolerance` (default 1e-8).
A backtracking line search reduces the nonlinear residual. Converged PV controls
that violate Q bounds become PQ at the bound and stay PQ during that run.

Sparse CSR storage replaces the original dense Y-bus/Jacobian. Default graph
ordering is Reverse Cuthill-McKee, then contiguous **paired real/imaginary rows**
are assigned to workers. `--ordering input` preserves caller ordering. This is
not an optimal geographical/METIS partitioner; random or highly meshed topology
may communicate more than a feeder-grouped model.

Default linear solving uses restarted GMRES and block-Jacobi ILU preconditioning.
`--threads N` executes native sparse row products and independent block solves
through a thread pool. Matrix assembly, Krylov reductions, parts of setup and
preconditioner factorization remain CPU work. `--linear direct` is a sparse-LU
reference mode, not a national-scale default. CG is not used for the generally
nonsymmetric NR Jacobian. Every Krylov solve checks the **true CPU linear residual**
and aborts on failure, even if worker/GPU results appear converged.

Useful controls:

```sh
python -m nrlarge solve --case my_case --out results --threads 8 --restart 30 --memory-gib 16 --tolerance 1e-9 --max-iterations 100
```

`--memory-gib` is a conservative model/assembly/Krylov preflight budget, not an OS
quota. ILU/LU fill, Python/runtime overhead, drivers, concurrent applications and
paging can exceed estimates. Reserve real RAM headroom. CSV ingestion streams
records and checks its budget before accumulating the full case; repeated stamp
conductor IDs share canonical strings. The coordinator still owns the full sparse
model and GMRES basis, approximately O(nnz + restart*unknowns), rather than fully
distributed graph storage.

## OpenCL and OpenGL numerical acceleration

```sh
python -m nrlarge hardware
python -m nrlarge solve --case my_case --out cl_results --threads 1 --accelerator opencl
python -m nrlarge solve --case my_case --out gl_results --threads 1 --accelerator opengl
```

Both backends execute **real FP64 CSR Jacobian-vector products on the GPU**.
Preconditioning, Krylov reductions, Newton decisions and final physical-residual
checks remain on CPU. Host/device transfers and context/shader setup are included
in solve timings and can make small cases slower. Compare end-to-end timings for
the intended model and hardware.
Explicit accelerator selection fails when unavailable; it never silently falls
back to CPU or labels a CPU OpenCL device/software renderer as GPU execution.

OpenCL uses the installed GPU ICD loader via ctypes and requires FP64 support;
no pyopencl package is needed. OpenGL requires PyOpenGL and OpenGL 4.3+ compute/FP64;
Windows uses a private hidden WGL context, while Linux uses optional GLFW. No
visible user window is created. Use `--threads 1` for OpenGL and one worker process
per GPU. OpenGL shard indices/buffers must fit the driver's uint32 and buffer/group
limits; OpenCL uses uint64 CSR indices. Different computers may use different
backends, each explicitly configured on its worker.

## Up to 20 mixed Windows/Linux workers

Workers perform **parts of one linear system**, not merely unrelated scenarios.
Each gets an owned CSR row interval, its compact neighbor-column map, and local
preconditioner block. The coordinator sends only needed vector entries and
assembles complete row results. Workers can use CPU threads or a local GPU.

There is no discovery, remote shell, automatic remote installation, shared pickle,
file-serving endpoint or listener opened by `solve`. Start workers deliberately on
trusted machines. Use the same source revision and compatible Python/NumPy/SciPy
on all machines. Configure firewalls outside this application.

Create a strong random token file locally and distribute it securely to the
coordinator and workers. Use at least 32 random ASCII characters. Protect it and
private keys with appropriate file permissions. Provision server TLS certificates
whose subject alternative names match each configured host, using your trusted CA.
The application does not create certificates, disable TLS verification or change
system security settings.

On a worker (same command form on Windows and Linux):

```sh
python -m nrlarge worker --bind 192.168.1.21 --port 8843 --token-file worker.token --cert worker.pem --key worker-key.pem --threads 8 --memory-gib 8
```

For a GPU worker, replace/add `--accelerator opencl` or `--accelerator opengl`
(OpenGL requires threads 1). Workers execute only authenticated numeric operations.
The service uses verified TLS 1.2+, constant-time token comparison, bounded frames,
finite timeouts, exclusive job leases, per-Jacobian epochs and ordered sequences.
Expired idle leases release matrices/contexts. Restart/reconnection is not silently
used to replace partial results; a failed worker fails the solve.

`workers.json` on the coordinator lists 1..20 unique HTTPS origins:

```json
["https://grid-worker-01.example:8843", "https://grid-worker-02.example:8843"]
```

```sh
python -m nrlarge solve --case my_case --out cluster_results --workers workers.json --token-file worker.token --ca-file cluster-ca.pem --memory-gib 32
```

The cluster accepts up to twenty computer/worker endpoints.
Within each endpoint, CPU row products can use its configured thread count;
preconditioner setup/solve is still a host sparse-factorization bottleneck.
Matrix uploads are chunked with exact coverage validation and RAM preflight;
individual messages and vector products remain bounded to 64 MiB. Transport is
JSON/base64 with explicit little-endian numeric types, so serialization/TLS
traffic may dominate strong scaling. The coordinator is a single point of failure.
Do not expose workers to untrusted clients or the public Internet.

For automated **local tests only**, HTTP can be enabled on loopback with
`--allow-loopback-http`; the option refuses remote HTTP endpoints/binds. Production
multi-computer use always requires verified TLS. No remote machine was contacted
in the current test evidence; separate local processes exercised the transport.

## CSV, ASCII, plots and customer drilldown

Reports preserve reloadable input CSVs and `case.json` under `inputs/`. Complete
outputs stream into 100,000-row CSV shards:

| Output | Meaning |
| --- | --- |
| `nodes-*.csv` | Every conductor voltage, actual source/net injection, phase and customer/feeder metadata |
| `loads-*.csv` | Every customer's terminal voltage and actual ZIP P/Q, in pu and MW/MVAr |
| `terminals-*.csv` | Per-element terminal currents/powers; sum ports for element complex loss |
| `feeders-*.csv` | Region/feeder sums of explicit customer-load consumption |
| `convergence-*.csv` | Nonlinear/current and power errors, linear iterations/residual/timing, damping |
| `summary.json`, `summary-*.csv` | Status, dimensions, base, losses, demand, execution configuration and resource estimate |
| `network.txt` | Full streamed ASCII element ports, load connections and conductor operating points |
| `voltage.svg`, `angles.svg`, `convergence.svg` | Bounded samples, with full values retained in CSV |

```sh
python -m nrlarge solve --case my_case --out focused_results --feeder F1 --plot-points 512
```

Feeder focus affects sampled charts, not full data exports. Plots are bounded to
2..4096 points and fixed canvas dimensions; they do not attempt to label millions
of customer nodes. Terminal powers are positive from a conductor into an element.
Input node injections are positive generation; explicit load powers are positive
consumption. Summaries use the three-phase base/3 conversion described above.

Full output can be larger than the model. `--max-output-mib` bounds its conservative
estimate (default 512 MiB). Exports estimated above 16 MiB are refused if the
destination would retain less than 100 GiB free. This is a per-operation preflight,
not an OS quota or a cross-session storage reservation service; large operations
near the reserve remain unsafe under concurrent writers. The program does not
relocate data, delete existing user files or install storage-management services.

## Capacity and verification

Synthetic in-memory studies avoid writing enormous fake datasets:

```sh
python -m nrlarge benchmark --customers 100000 --threads 4 --memory-gib 3 --tolerance 1e-10
```

See [NRlarge verification](NRlarge-verification.md) for actual measured results,
platform/device details, tests and untested deployment coverage. The synthetic
case uses many short, lightly loaded feeders tied to ideal fixed-voltage source
conductors. It exercises structural capacity for that topology. Strong/weak
scaling and regional transmission/distribution models have separate entries in
the verification record.

For state/regional-grid qualification, supply a versioned real HV/MV/LV/customer
model with validated voltage bases, transformer/neutral/grounding primitives,
customer phase and connection assignments, load models and operating points.
Then compare phase voltages, power balances and limit behavior to independent
reference tools, measure memory/fill and end-to-end scaling on the actual 10..20
machines, and test worker/device failure. No optimal power flow, dynamic/transient
simulation, protection coordination, automatic tap/regulator control, frequency
stability is included. True distributed
model storage and mature MPI/PETSc/Krylov/preconditioner transport are the next
architecture step if coordinator/JSON limitations bind.

Primary numerical/runtime references: [SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html),
[PETSc distributed networks](https://petsc.org/release/manual/dmnetwork/),
[OpenCL specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html),
[OpenGL 4.3 specification](https://registry.khronos.org/OpenGL/specs/gl/glspec43.core.pdf).
