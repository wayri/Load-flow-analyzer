# Load flow analyzer

A portable C++17 console application for balanced, steady-state AC load-flow studies.
Originally written by Yawar Badri in 2019; the original implementation remains in Git history.

- **Gauss_Siedel_Load_Flow_Analyser** branch defaults to Gauss-Seidel.
- **Newton_Raphson** branch defaults to damped Newton-Raphson.
- Both methods remain selectable on either branch with `--method gs` or `--method nr`.
- Import and save bus/branch input CSVs; export solved bus powers, branch flows/losses,
  convergence history, ASCII diagrams, and standalone SVG plots.
- SLACK, PQ and PV buses; per-bus reactive limits; line charging, parallel branches,
  off-nominal transformer taps and phase shifts; automatic Y-bus construction.

## Build and run

Requires a C++17 compiler and CMake 3.16 or newer. No third-party libraries or plotting
runtime are required. Use Visual Studio 2022, current GCC, or Clang.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Linux/macOS or a single-configuration Windows build:

```sh
./build/load_flow --buses examples/three_bus/buses.csv --branches examples/three_bus/branches.csv --method nr --out results
```

With Visual Studio's CMake generator, the executable is `build/Release/load_flow.exe`.
The existing `.sln` also targets Visual Studio 2022/C++17; CMake is the portable build.
For a quick GCC build without CMake:

```sh
g++ -std=c++17 -O2 Gauss_Siedel_Load_Flow_Analyser/Gauss_Siedel_Load_Flow_Analyser.cpp -o load_flow
```

Run `load_flow --help` for options. `load_flow --interactive` prompts for CSV paths,
method and report directory. To create editable inputs:

```sh
load_flow --template my_case
load_flow --buses my_case/input_buses.csv --branches my_case/input_branches.csv --out my_results
```

Edit the template in a text editor or spreadsheet. Reports require a new or empty
directory so existing files are preserved. No arguments prints help. Exit status is
**0** for success, **1** for invalid input/I/O/numerical errors, and **2** for a solve
that did not converge. Nonconverged reports contain the last iterate, explicitly
marked `converged=0`; do not treat them as a solved operating point.

## CSV contract

Use UTF-8 (optional BOM), comma separators and a dot decimal separator. Headers are
required in the exact order below. LF and CRLF, blank lines, quoted commas, escaped
quotes and quoted multiline fields are supported. Surrounding whitespace is trimmed.
IDs must be unique within each table; branch endpoints reference bus IDs. Bus types
are case-sensitive. All numeric fields must be finite; only Q limits may be blank.

`buses.csv`:

```csv
id,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu
1,SLACK,0,0,1,0,,
2,PQ,-0.5,-0.45,1,0,,
```

| Column | Meaning |
| --- | --- |
| `id` | Nonempty bus identifier |
| `type` | `SLACK`, `PQ`, or `PV`; exactly one SLACK |
| `p_pu`, `q_pu` | Net bus power: generation positive, demand negative |
| `vm_pu` | Positive initial voltage; fixed setpoint for SLACK/PV |
| `va_deg` | Initial angle; fixed reference angle for SLACK |
| `qmin_pu`, `qmax_pu` | PV net reactive-injection limits; blank means unbounded |

SLACK P/Q and PV Q inputs are placeholders, not controlled quantities. PQ voltages
are initial guesses. If a generator shares a bus with demand, enter net P/Q and
subtract local reactive demand from its generator Q limits. Outputs report net
injections; recovering generator output requires adding local demand back.

`branches.csv`:

```csv
id,from,to,r_pu,x_pu,b_pu,tap,shift_deg
L1,1,2,0,0.1,0,1,0
```

| Column | Meaning |
| --- | --- |
| `id`, `from`, `to` | Branch ID and distinct endpoint bus IDs |
| `r_pu`, `x_pu` | Series impedance; R >= 0, nonzero impedance |
| `b_pu` | Total shunt charging susceptance, split equally between ends |
| `tap` | Positive magnitude of from-side transformer tap; 1 for a line |
| `shift_deg` | From-side tap phase angle; 0 for a line |

All data must use **one common per-unit base**. Convert MW/MVAr by dividing by base
MVA; convert impedances consistently before import. The program does not infer a
base or mix physical and per-unit values. A header-only branch table is allowed
for a single SLACK bus. Islands, missing endpoints and zero-impedance branches are
rejected. Parallel branches use separate IDs.

## Reports and diagrams

| File | Contents |
| --- | --- |
| `input_buses.csv`, `input_branches.csv` | Normalized, reloadable input data |
| `buses.csv` | Requested/solved bus type, voltage, actual net P/Q injections, status |
| `branches.csv` | P/Q at both terminals, complex loss, status |
| `convergence.csv` | Maximum absolute P/Q mismatch at each iteration |
| `summary.csv` | Method, status, update count, residual, total branch P/Q loss |
| `network.txt` | Complete ASCII branch connectivity and bus operating points |
| `voltage.svg`, `angles.svg`, `convergence.svg` | Standalone plots; open in a browser |

Branch terminal powers are **positive from the bus into the branch**. Adding both
terminal powers yields branch loss. Reactive loss can be negative because line
charging supplies vars. CSV power values remain in per unit. Iteration 0 is the
initial state; the final history row is the returned state.

Example network representation (each branch once; shared IDs denote shared nodes):

```text
[1]----{L12}----[2]
[1]----{L13}----[3]
[2]----{L23}----[3]
```

The generated diagram includes impedance, charging, tap and phase shift on every
branch, followed by voltage and actual net injections for every bus. This edge-based
representation retains loops and parallel connections without ambiguous crossings.

```text
bus CSV + branch CSV
         |
         v
validate IDs, units contract, types and connectivity
         |
         v
construct complex Y-bus
         |
         +--> Gauss-Seidel sequential updates --+
         |                                     |
         +--> Newton-Raphson Jacobian + damping +--> power mismatch check
                                                        |
                                    enforce PV Q limits; repeat if changed
                                                        |
                                                        v
                                      bus powers + terminal flows + losses
                                                        |
                                                        v
                                      CSV + ASCII network + SVG plots
```

## Numerical method and limits

Newton-Raphson solves non-slack P and PQ Q balance equations in polar coordinates
using an analytical Jacobian recomputed each iteration, partial-pivoted linear solve,
and backtracking to reduce mismatch while preserving positive voltage magnitudes.
This is an established modern power-flow method, not a claim of a novel or globally
convergent algorithm. See the [MATPOWER AC power-flow formulation](https://matpower.app/manual/matpower/ACPowerFlow.html).
Gauss-Seidel uses the latest available voltages within each sweep and restores PV
voltage magnitude after each update. Both use the same power-mismatch criterion.

PV Q limits are checked after the current bus-type system converges. Violating
buses switch to PQ at the nearest bound and remain PQ for the rest of the run;
voltage magnitude may then depart from its setpoint. The iteration budget includes
these extra updates. SLACK Q limits are not enforced.

Default tolerance is `1e-8 pu`, with 100 updates; use `--tolerance` and
`--max-iterations` to adjust. Ill-conditioned or infeasible cases may fail or not
converge. The dense implementation is capped at 500 buses and intended for small
engineering/educational studies. It is not a sparse large-grid solver, an OPF,
three-phase unbalanced solver, contingency engine, automatic tap controller or
voltage/thermal-limit checker. Independent engineering validation is required for
operational use.

## Verification and migration

Tests cover an exact two-bus phasor solution, rotated slack reference, PV regulation,
upper/lower Q limits, the official MATPOWER case9 network, GS/NR agreement, branch
power balance, transformer stamps, invalid/disconnected networks, nonconvergence,
CSV round trips, malformed quotes, UTF-8 BOM/CRLF and escaped SVG labels.
Case9 source provenance is recorded in `tests/test_solver.cpp`; the fixture tests
power equations and losses rather than claiming an external solver was run locally.

The old `bam`, `tbus`, `pbus` and `GSLF` prompt commands are replaced by reproducible
CSV cases and explicit CLI options. Branch data now generates Y-bus internally;
raw Y-bus import is not supported. Original source is available in the repository's
pre-modernization history.
