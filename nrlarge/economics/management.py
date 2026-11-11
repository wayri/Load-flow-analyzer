"""Explicit projected state and rolling replans at existing interval boundaries.

after_slot counts completed slots: zero is the opening state and T is the
closing state. Projected state is never relabelled as measured telemetry.
"""
from copy import deepcopy
from datetime import timedelta
import math

from .model import instant


_TOL = 1e-6


def baseline_case(case):
    """Same assets, obligations, prices and limits, with only storage removed."""
    result = deepcopy(case)
    result.batteries = []
    return result.validate()


def _grid(case):
    rows = next(iter(case.periods.values()))
    return rows, [instant(row.time) for row in rows]


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def _range(value, low, high, name):
    value = _number(value, name, minimum=None)
    if value < low-_TOL or value > high+_TOL:
        raise ValueError(f"{name} is outside [{low}, {high}]")
    # Remove only sub-tolerance solver/serialization roundoff, not measurement
    # discrepancies large enough to alter the physical planning contract.
    return min(high, max(low, value))


def _group(state, name, assets):
    value = state.get(name, {})
    expected = {asset.id for asset in assets}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"State {name} IDs must match the case exactly")
    if any(not isinstance(row, dict) for row in value.values()):
        raise ValueError(f"State {name} values must be objects")
    return value


def _fields(row, required, optional=()):
    if not set(required) <= set(row) or set(row)-set(required)-set(optional):
        raise ValueError(f"State entry requires {', '.join(required)}; unknown fields are not allowed")


def state_at(case, dispatch, after_slot):
    """Project a replay-validated schedule's complete managed-asset state."""
    case.validate()
    rows, _ = _grid(case)
    if type(after_slot) is not int or not 0 <= after_slot <= len(rows):
        raise ValueError("after_slot is a count of completed slots from 0 through T")
    from .validation import validate_dispatch
    replay = validate_dispatch(case, dispatch)
    if not dispatch.feasible or not replay["valid"]:
        raise ValueError("Cannot project state from an infeasible or invalid dispatch")
    boundary = (rows[after_slot].time if after_slot < len(rows) else
                (instant(rows[-1].time)+timedelta(hours=rows[-1].duration_h)).isoformat())
    state = {"schema": 1, "boundary": boundary, "source": "projected",
             "after_slot": after_slot, "batteries": {}, "generators": {},
             "sites": {}, "flexible_loads": {}}
    durations = [row.duration_h for row in rows[:after_slot]]
    for battery in case.batteries:
        values = dispatch.batteries[battery.id]
        hi = battery.capacity_mwh if battery.soc_max_mwh is None else battery.soc_max_mwh
        soc = _range(values["energy_mwh"][after_slot], battery.soc_min_mwh, hi,
                     battery.id+" projected SOC")
        remaining = None
        if battery.max_discharge_mwh is not None:
            delivered = sum(power*duration for power, duration in
                            zip(values["discharge_mw"][:after_slot], durations))
            remaining = _range(battery.max_discharge_mwh-delivered, 0,
                               battery.max_discharge_mwh, battery.id+" remaining discharge budget")
        state["batteries"][battery.id] = {"soc_mwh": soc,
                                          "remaining_discharge_mwh": remaining}
    for generator in case.generators:
        values = dispatch.generators[generator.id]
        if after_slot == 0:
            on, power, elapsed = generator.initial_on, generator.initial_p_mw, generator.initial_state_h
        else:
            on = int(round(values["on"][after_slot-1]))
            power = _range(values["p_mw"][after_slot-1], generator.p_min_mw*on,
                           generator.p_max_mw*on, generator.id+" projected output")
            elapsed = 0
            for slot in range(after_slot-1, -1, -1):
                if int(round(values["on"][slot])) != on:
                    break
                elapsed += rows[slot].duration_h
            else:
                if on == generator.initial_on:
                    elapsed += generator.initial_state_h
        state["generators"][generator.id] = {"on": int(on), "p_mw": float(power),
                                             "status_duration_h": float(elapsed)}
    for site in case.sites:
        prior = dispatch.site[site.id]["import_mw"][:after_slot]
        state["sites"][site.id] = {"historical_peak_mw": float(max([site.historical_peak_mw, *prior]))}
    for load in case.flexible_loads:
        served = sum(power*duration for power, duration in
                     zip(dispatch.flexible[load.id]["p_mw"][:after_slot], durations))
        remaining = _range(load.energy_mwh-served, 0, load.energy_mwh,
                           load.id+" remaining energy obligation")
        state["flexible_loads"][load.id] = {"remaining_energy_mwh": remaining}
    return state


def replan(case, state, from_time):
    """Clip forecasts and carry explicit measured/projected asset obligations.

    The returned case retains the original terminal SOC commitment, including
    when the original terminal target defaulted to its original opening SOC.
    fixed_horizon_cost remains the supplied forecast's fixed fee; this function
    does not infer which fees have been incurred or build a realized cash ledger.
    """
    case.validate()
    allowed = {"schema", "boundary", "source", "after_slot", "batteries",
               "generators", "sites", "flexible_loads"}
    if (not isinstance(state, dict) or set(state)-allowed
            or type(state.get("schema")) is not int or state["schema"] != 1):
        raise ValueError("Require schema-1 managed state without unknown fields")
    if state.get("source") not in ("measured", "projected"):
        raise ValueError("State source must explicitly be measured or projected")
    if not isinstance(state.get("boundary"), str) or not isinstance(from_time, str):
        raise ValueError("State boundary and from_time must be timezone-aware timestamps")
    boundary = instant(state["boundary"])
    if instant(from_time) != boundary:
        raise ValueError("from_time must equal the explicit state boundary")
    rows, grid = _grid(case)
    try:
        start = grid.index(boundary)
    except ValueError as error:
        raise ValueError("Replan boundary must be an existing interval start with time remaining") from error
    # This count is provenance from the previous forecast. A replacement
    # forecast can start at boundary and therefore use a different local index.
    if "after_slot" in state and (type(state["after_slot"]) is not int or state["after_slot"] < 0):
        raise ValueError("State after_slot must be a nonnegative completed-slot count")
    batteries = _group(state, "batteries", case.batteries)
    generators = _group(state, "generators", case.generators)
    sites = _group(state, "sites", case.sites)
    flexible = _group(state, "flexible_loads", case.flexible_loads)
    result = deepcopy(case)
    result.periods = {site: periods[start:] for site, periods in result.periods.items()}
    for original, battery in zip(case.batteries, result.batteries):
        entry = batteries[battery.id]
        _fields(entry, ("soc_mwh",), ("remaining_discharge_mwh",))
        hi = original.capacity_mwh if original.soc_max_mwh is None else original.soc_max_mwh
        battery.initial_soc_mwh = _range(entry["soc_mwh"], original.soc_min_mwh, hi,
                                         battery.id+" measured SOC")
        battery.terminal_soc_mwh = (original.initial_soc_mwh if original.terminal_soc_mwh is None
                                   else original.terminal_soc_mwh)
        if original.max_discharge_mwh is not None:
            if "remaining_discharge_mwh" not in entry:
                raise ValueError(f"{battery.id}: remaining discharge budget is required")
            battery.max_discharge_mwh = _range(entry["remaining_discharge_mwh"], 0,
                original.max_discharge_mwh, battery.id+" remaining discharge budget")
        elif entry.get("remaining_discharge_mwh") is not None:
            raise ValueError(f"{battery.id}: case has no discharge-energy budget")
    for generator in result.generators:
        entry = generators[generator.id]
        _fields(entry, ("on", "p_mw", "status_duration_h"))
        if type(entry["on"]) not in (int, bool) or entry["on"] not in (0, 1):
            raise ValueError(f"{generator.id}: on must be 0 or 1")
        generator.initial_on = int(entry["on"])
        generator.initial_p_mw = _range(entry["p_mw"], generator.p_min_mw*generator.initial_on,
            generator.p_max_mw*generator.initial_on, generator.id+" measured output")
        generator.initial_state_h = _number(entry["status_duration_h"], generator.id+" status duration")
    for original, site in zip(case.sites, result.sites):
        entry = sites[site.id]
        _fields(entry, ("historical_peak_mw",))
        value = _number(entry["historical_peak_mw"], site.id+" historical peak")
        if value < original.historical_peak_mw-_TOL:
            raise ValueError(f"{site.id}: historical peak cannot decrease within this charge horizon")
        site.historical_peak_mw = max(value, original.historical_peak_mw)
    remaining_durations = [row.duration_h for row in next(iter(result.periods.values()))]
    for original, load in zip(case.flexible_loads, result.flexible_loads):
        entry = flexible[load.id]
        _fields(entry, ("remaining_energy_mwh",))
        load.energy_mwh = _range(entry["remaining_energy_mwh"], 0, original.energy_mwh,
                                 load.id+" remaining energy")
        old_end = len(rows) if original.end_slot is None else original.end_slot
        load.start_slot = max(0, original.start_slot-start)
        load.end_slot = max(0, old_end-start)
        possible = load.p_max_mw*sum(remaining_durations[load.start_slot:load.end_slot])
        if load.energy_mwh > possible+_TOL:
            raise ValueError(f"{load.id}: remaining flexible energy cannot fit its remaining window")
    return result.validate()
