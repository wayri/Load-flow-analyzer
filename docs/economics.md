# Economic planning and dispatch

The `NRlarge` branch includes a market-neutral, multi-site scheduling module for
battery stations, generators, distribution companies and aggregators. It chooses
interval dispatch to maximize modeled operating value over your forecast horizon.
Use the supplied CSVs for prices, forecasts, asset limits and costs.

## Run a battery station

Use the existing Python environment with NumPy, SciPy and the plotting dependencies
listed in `requirements-benchmarks.txt`. SciPy 1.14+ supplies the HiGHS
mixed-integer optimizer. Windows and Linux use the same commands from the repo root.

```sh
python -m nrlarge economics example --role battery --out my-battery
python -m nrlarge economics optimize --case my-battery --out day-ahead --baseline
python -m nrlarge economics check --case day-ahead/inputs --result day-ahead
```

Edit `my-battery/periods.csv`, `sites.csv` and `batteries.csv` before optimizing your
own station. `case.json` holds the complete case; each existing CSV overrides its
corresponding JSON table. Deleting a CSV restores that table's JSON values.
`generators.csv` and `flexible_loads.csv` work the same way. Empty tables retain
their headers. Fields, column order and unknown IDs are checked on import.

Each result directory contains:

- `inputs/`: reloadable snapshot of the forecast and asset configuration used.
- `dispatch.json`: portable full schedule, solver status and accounting receipt.
- `dispatch.csv`: import, export, renewable use and demand per meter and interval.
- `battery_soc.csv`: charge/discharge and T+1 stored-energy boundary states.
- `generator_dispatch.csv` and `flexible_dispatch.csv`: asset schedules.
- `cashflow.csv` and `horizon_charges.csv`: interval cash flows and separate fees.
- `summary.json`: objective, independent replay, baseline comparison and AC check.
- PNG/SVG plots: dispatch, state of charge, prices and cumulative cash.
- `topology.txt`: ASCII overview of sites and their assets.

`check` reconstructs power balance, stored energy, operating modes, generator
transitions/ramps, flexible-load obligations and all cash flows from the saved
schedule and inputs. It also compares the saved monetary receipt with its own
calculation. It returns exit code 2 for invalid schedules. Optimizer commands return
0 for feasible schedules, 2 for infeasible/invalid schedules or failed AC checks,
and 1 for invalid input or file errors. Output directories must be new or empty;
existing state/check JSON files are preserved.

## Plan and manage dispatch

Use `optimize` for a day-ahead schedule. At each operating boundary, update
forecasts and measured state, then call `replan` for the remaining intervals.

```sh
python -m nrlarge economics state --case day-ahead/inputs --result day-ahead --after-slot 6 --out state.json
python -m nrlarge economics replan --case my-battery --state state.json --out intraday --baseline
python -m nrlarge economics check --case intraday/inputs --result intraday
```

`after-slot 6` means six completed intervals. The exported state is explicitly
`"source": "projected"`, suitable for a simulated replay. For operation, replace
the values with measurements and set `"source": "measured"`. The mandatory
`boundary` is an ISO8601 timestamp matching an interval start with time remaining.
`--from-time` may specify that same timestamp explicitly. Future intervals can be
edited before replanning; retain original timestamps and service windows when
using the original forecast. Replacement forecasts must describe their own
remaining flexible-load windows.

The state includes battery stored energy and remaining discharge budget;
generator on/off state, output and time in that state; remaining flexible-load
energy; and each meter's historical billed peak. Replanning carries these
obligations and preserves the original terminal SOC commitment. Adjust any fixed
horizon fee in the supplied forecast to reflect the costs being assessed in that
replan. Each receipt evaluates its own remaining horizon; recorded actual meter
energy and settlement prices are needed to assess realized cash flows.

The CSV schedule is a planning/dispatch instruction artifact. The module runs on
demand; connecting it to a station controller, telemetry system or market bidding
service requires an application-specific integration.

### Compare battery capacities and power ratings

Create a CSV such as:

```csv
candidate_id,battery_id,capacity_mwh,charge_max_mw,discharge_max_mw,horizon_fixed_cost
none,B1,0,0,0,0
small,B1,3,1,1,5
reference,B1,6,2,2,10
large,B1,9,3,3,20
```

```sh
python -m nrlarge economics plan --case my-battery --candidates candidates.csv --out sizing
```

Each row changes one existing battery. Other assets remain in the portfolio.
SOC minimum/maximum, initial/terminal energy and discharge budget scale with
capacity, preserving fractions. The reference battery must have cyclic initial
and terminal SOC. Zero capacity removes it. `horizon_fixed_cost` is an additional
user-supplied cost charged once in the modeled horizon, on top of the case fee.
Use matching cost assumptions across candidates. The output ranks feasible
candidates by total horizon cost and retains their schedules and input snapshots.
Up to 50 candidates can be compared; each gets its own solve time limit.

## Model and units

All site intervals share explicit timezone-aware, contiguous boundaries.
Duration is in hours, power in MW, energy in MWh, and energy prices in the
configured currency/MWh. Demand charges are currency/MW **over this supplied
charge horizon**. Use `historical_peak_mw` to carry the already billed peak.
For a partial billing month, forecast the remaining month or account explicitly
for future peaks when interpreting the schedule. Each case uses one currency.

At every site and interval:

```text
                         +---- generator
                         +---- renewable used
grid import ---> [PCC] --+---- battery (charge / discharge)
grid export <---   |     +---- fixed demand + flexible demand
                  +---------- forecast network losses

import + generation + renewable + discharge
    = export + fixed load + flexible load + charge + forecast losses
```

Meters have separate import/export caps and mutually exclusive import/export
modes. Portfolio profit sums site cash flows. A portfolio can import at one
meter and export at another simultaneously.

Battery energy follows
`E_next = (1-self_discharge_per_h)^duration * E + eta_charge*charge*duration - discharge*duration/eta_discharge`.
Charge/discharge modes are exclusive. SOC limits, power limits, initial energy,
exact terminal energy and optional delivered-discharge budget are enforced.
An omitted terminal target restores the initial energy. Degradation cost is
charged per **delivered AC discharge MWh**, with a constant efficiency/cost model.

Generator costs include marginal energy cost, online cost and startup cost.
Binary commitment, minimum output, startup/shutdown ramps, ordinary ramp rates,
minimum up/down hours and initial operating state are enforced. A transition
near the horizon end carries any remaining minimum-time obligation into the
exported state. Flexible loads have exact energy requirements, maximum power
and `[start_slot,end_slot)` service windows. Fixed demand must be served.

The objective minimizes net cost: purchases + generation + startup + degradation
+ curtailment penalty + incremental demand charge + fixed fee − export revenue
− fixed-load retail revenue − flexible-load service value. Net profit is its
negative. Prices and generator marginal costs may be negative. Caps and exclusive
modes bound the physical transactions. Fixed retail/service values are input
accounting assumptions; they do not imply an operating margin by themselves.

`--baseline` reoptimizes the same portfolio with all batteries removed. The
reported benefit is baseline cost minus dispatch cost. An infeasible baseline
retains its status and produces no monetary comparison. A time-limited feasible
incumbent is labelled `feasible_limit`; the solver gap/status remain in the
receipt. An optimal result meets the configured MILP tolerances. The API exposes
`time_limit` and `mip_rel_gap` (default 60 seconds and 1e-8).

The optimizer is a single-process sparse MILP with budgets of 100,000 variables,
500,000 constraints and 2,000,000 nonzero coefficients. Use aggregated commercial
assets and appropriately sized horizons. NRlarge's distributed/GPU AC engine is
available for network studies; this economic optimizer uses CPU HiGHS.

## Optional unbalanced AC replay

`optimize` and `replan` accept `--grid-mapping mapping.json`. Each meter maps to a
passive NRlarge conductor network with SLACK sources, PQ conductors, zero existing
node injections, and no existing loads. Schedule resources are added once as
constant-power phase-to-neutral devices. The network's three-phase MVA base is
converted to conductor base by dividing by three.

```json
{
  "sites": {
    "battery-meter": {
      "case": "passive-feeder",
      "voltage_min_pu": 0.95,
      "voltage_max_pu": 1.05,
      "interchange_tolerance_mw": 0.001,
      "terminal_current_limits_pu": {"line1": 1.2},
      "ports": {
        "fixed_load": [{"from":"load.A","to":"load.N","weight":1,"q_per_p":0.3}],
        "renewable": [{"from":"load.A","to":"load.N","weight":1,"q_per_p":0}],
        "battery:B1": [{"from":"load.A","to":"load.N","weight":1,"q_per_p":0}]
      }
    }
  }
}
```

Case paths are relative to the mapping file. Each resource's weights sum to one;
use three ports to split balanced power across phases. Map every generator as
`generator:ID` and flexible load as `flexible:ID`. Voltage limits apply to mapped
phase-to-neutral port voltages on a nominal 1 pu basis. Element current ratings
bound the maximum absolute conductor-terminal current from its admittance stamp.
Configure ratings for all elements whose thermal limits you want checked.

The replay reports AC convergence, port voltages, configured terminal ratings,
actual PCC interchange and its difference from scheduled interchange. Actual AC
losses replace the forecast loss allowance; adding forecast losses as loads
would count them twice. Its pass flag requires all configured checks, including
PCC agreement within the supplied tolerance. An omitted grid check is recorded
as unchecked. Economic scheduling and AC replay have separate feasibility
results. Network constraints are evaluated after scheduling; revise loss
forecasts, caps or dispatch assumptions and reoptimize as needed.

## Examples, evidence and references

The [published examples](../economics/README.md) include four synthetic role
cases, full CSV/JSON results and plots, plus replanning and sizing examples.
The numerical suite compares against hand-calculated cases and an independent
enumeration of small binary-mode linear programs:

```sh
python -B -m unittest discover -s tests/economics -v
```

The solver interface and termination states follow [SciPy's MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
See [acknowledgements, disclaimer and waiver](../ACKNOWLEDGEMENTS.md).
