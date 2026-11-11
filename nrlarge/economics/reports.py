"""CSV cash/schedule evidence and bounded dispatch plots; no fabricated schedules."""
from dataclasses import asdict, is_dataclass
from copy import copy
from io import BytesIO
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _series(value, name, count):
    result = np.asarray(value[name], dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"Report requires {count} finite values for {name}")
    return result


def _replayed(case, dispatch, label):
    """Replay public arrays and use canonical derived money before any export."""
    from .validation import validate_dispatch
    replay = validate_dispatch(case, dispatch)
    if not replay["valid"]:
        raise ValueError(f"{label} failed independent replay: " + "; ".join(replay["violations"][:5]))
    if dispatch.validation.get("valid") is False:
        raise ValueError(f"{label} carries a failed validation receipt")
    fresh = replay["accounting"]
    if dispatch.accounting.get("currency", case.currency) != case.currency:
        raise ValueError(f"{label} accounting currency disagrees with the input case")
    for key in ("total_cost", "net_profit"):
        if key in dispatch.accounting:
            saved = float(dispatch.accounting[key])
            if not math.isfinite(saved) or not math.isclose(saved, fresh[key], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError(f"{label} saved {key} disagrees with independent replay")
    if dispatch.objective_cost is not None and not math.isclose(float(dispatch.objective_cost), fresh["total_cost"], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"{label} objective cost disagrees with independent replay")
    result = copy(dispatch)
    result.accounting = fresh
    result.validation = replay
    result.objective_cost = fresh["total_cost"]
    return result


def _cash(case, result):
    """Interval cash followed by reconciled demand/fixed charges at horizon end."""
    accounting = _get(result, "accounting", {})
    count = len(next(iter(case.periods.values())))
    cash = np.zeros(count)
    for line in accounting.get("lines", []):
        slot = int(line["slot"])
        if not 0 <= slot < count or not math.isfinite(float(line["net_profit"])):
            raise ValueError("Invalid cashflow slot/value in accounting evidence")
        cash[slot] += float(line["net_profit"])
    cumulative = np.r_[0.0, np.cumsum(cash)]
    final = float(accounting["net_profit"])
    return np.r_[cumulative, final], final-cumulative[-1]


def _topology(case):
    def label(value):
        return str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    lines = [f"{case.role.upper()} ASSET OVERVIEW", "Each PCC has its own meter balance in the dispatch model.", ""]
    for site in case.sites:
        rows = case.periods[site.id]
        lines.append(f"[Grid market] <--> [PCC: {label(site.id)}]")
        lines.append(f"                    Meter limits: import {site.import_limit_mw:g} MW / export {site.export_limit_mw:g} MW")
        devices = [f"Fixed load + supplied losses: {min(p.load_mw+p.loss_mw for p in rows):g}..{max(p.load_mw+p.loss_mw for p in rows):g} MW",
                   f"Renewable availability: {min(p.renewable_mw for p in rows):g}..{max(p.renewable_mw for p in rows):g} MW"]
        devices += [f"Battery {label(a.id)}: {a.capacity_mwh:g} MWh; charge {a.charge_max_mw:g} MW / discharge {a.discharge_max_mw:g} MW"
                    for a in case.batteries if a.site == site.id]
        devices += [f"Generator {label(a.id)}: {a.p_min_mw:g}..{a.p_max_mw:g} MW; commitment + ramp limits"
                    for a in case.generators if a.site == site.id]
        devices += [f"Flexible load {label(a.id)}: {a.energy_mwh:g} MWh obligation; {a.p_max_mw:g} MW maximum"
                    for a in case.flexible_loads if a.site == site.id]
        for k, device in enumerate(devices):
            lines.append("                    " + ("'-- " if k == len(devices)-1 else "|-- ") + device)
        lines.append("")
    return "\n".join(lines).rstrip()+"\n"


def _plots(case, result, directory, baseline):
    # Imported only for feasible schedules; an infeasible run emits no fake plots
    # and does not need a graphics runtime.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".task-check" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    first = next(iter(case.periods.values()))
    count = len(first)
    edges = np.r_[0.0, np.cumsum([p.duration_h for p in first])]
    currency = str(case.currency)
    files = {}
    battery_palette = ("#9058ac", "#25999e", "#9a6841", "#bd4f88", "#849334")
    generator_palette = ("#b84960", "#7756a1", "#a16b2e")
    flexible_palette = ("#657c88", "#838b35", "#885d80")
    battery_colors = {a.id:battery_palette[k % len(battery_palette)] for k,a in enumerate(case.batteries)}
    generator_colors = {a.id:generator_palette[k % len(generator_palette)] for k,a in enumerate(case.generators)}
    flexible_colors = {a.id:flexible_palette[k % len(flexible_palette)] for k,a in enumerate(case.flexible_loads)}
    style = {"font.family":"DejaVu Sans", "font.size":9, "text.color":"#20334a",
             "axes.labelcolor":"#20334a", "axes.edgecolor":"#c5cfda",
             "axes.spines.top":False, "axes.spines.right":False,
             "xtick.color":"#64748b", "ytick.color":"#64748b",
             "grid.color":"#e5eaf0", "grid.linewidth":.65,
             "svg.fonttype":"none", "svg.hashsalt":"nrlarge-economics-v1",
             "figure.facecolor":"white", "savefig.facecolor":"white"}

    def header(fig, name, note=""):
        fig.suptitle(f"{case.role.capitalize()} · {name}", x=.09, y=.97,
                     ha="left", fontsize=16, fontweight="bold")
        status = "OPTIMAL" if result.optimal else "FEASIBLE · OPTIMALITY NOT ESTABLISHED"
        fig.text(.09, .914, f"{count} intervals · {edges[-1]:g} hours · {len(case.sites)} meter site(s) · {currency} · {status}",
                 fontsize=8, color="#64748b")
        if note:
            fig.text(.09, .025, note, fontsize=7.5, color="#64748b")

    def save(fig, stem):
        svg = BytesIO()
        fig.savefig(svg, format="svg", metadata={"Date":None, "Creator":"NRlarge economics"})
        # Matplotlib path segments contain redundant line-end spaces. Emit a
        # consistent text artifact while retaining the path's newline separators.
        (directory/f"{stem}.svg").write_bytes(('\n'.join(line.rstrip() for line in svg.getvalue().decode('utf-8').splitlines())+'\n').encode('utf-8'))
        svg.close()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        with Image.open(buffer) as image:
            image.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT,
                                           dither=Image.Dither.NONE).save(directory/f"{stem}.png", optimize=True, compress_level=9)
        buffer.close()
        plt.close(fig)
        files[f"{stem}_svg"] = f"{stem}.svg"
        files[f"{stem}_png"] = f"{stem}.png"

    def step(axis, values, label, color=None, **kwargs):
        values = np.asarray(values, dtype=float)
        if len(values) != count:
            raise ValueError("Plot series does not match interval count")
        # Bound SVG geometry independently of CSV horizon length, retaining the
        # minimum and maximum of each displayed series.
        if count > 512:
            keep = np.unique(np.r_[np.linspace(0, count-1, 510, dtype=int),
                                   np.argmin(values), np.argmax(values)])
        else:
            keep = np.arange(count)
        x = np.r_[edges[keep], edges[-1]]
        y = np.r_[values[keep], values[-1]]
        axis.step(x, y, where="post", label=label, color=color, linewidth=1.35, **kwargs)

    with matplotlib.rc_context(style):
        # All sites in the supplied examples have their own readable subplot.
        # Larger fleets are split into pages instead of squeezing panels away.
        for page, start in enumerate(range(0, len(case.sites), 3)):
            sites = case.sites[start:start+3]
            fig, axes = plt.subplots(len(sites), 1, figsize=(11.2, 4.9+1.25*(len(sites)-1)), squeeze=False)
            fig.subplots_adjust(left=.09, right=.97, bottom=.14, top=.82, hspace=.75)
            header(fig, "meter dispatch", "Imports and sources are positive; grid export and battery charging are negative. Demand is a positive reference curve.")
            for axis, site in zip(axes[:, 0], sites):
                rows = case.periods[site.id]
                values = result.site[site.id]
                step(axis, _series(values, "import_mw", count), "Grid import", "#1479b8")
                step(axis, -_series(values, "export_mw", count), "Grid export (−)", "#dc7a22")
                demand = np.array([p.load_mw+p.loss_mw for p in rows])
                if np.any(demand):
                    step(axis, demand, "Base demand + supplied losses", "#172b42", linestyle="--")
                renewable = _series(values, "renewable_used_mw", count)
                if np.any([p.renewable_mw for p in rows]):
                    step(axis, renewable, "Renewable used", "#238667")
                for asset in case.batteries:
                    if asset.site == site.id:
                        values = result.batteries[asset.id]
                        step(axis, _series(values, "discharge_mw", count)-_series(values, "charge_mw", count),
                             f"Battery {asset.id}: net MW", battery_colors[asset.id])
                for asset in case.generators:
                    if asset.site == site.id:
                        step(axis, _series(result.generators[asset.id], "p_mw", count), f"Generator {asset.id}", generator_colors[asset.id])
                for asset in case.flexible_loads:
                    if asset.site == site.id:
                        step(axis, _series(result.flexible[asset.id], "p_mw", count), f"Flexible {asset.id}", flexible_colors[asset.id], linestyle=":")
                axis.set_title(site.id, loc="left", fontsize=11, fontweight="bold")
                axis.set_ylabel("Power (MW)")
                axis.set_xlabel("Elapsed horizon time (h)")
                axis.axhline(0, color="#9ba9b8", linewidth=.65)
                axis.grid(axis="y")
                axis.legend(ncol=3, loc="best", frameon=False, fontsize=7)
            save(fig, "dispatch" if page == 0 else f"dispatch-{page+1}")
        fig, axis = plt.subplots(figsize=(11.2, 5.3))
        fig.subplots_adjust(left=.09, right=.97, bottom=.16, top=.82)
        header(fig, "battery stored energy", "Energy is shown at interval boundaries; the final state is not an extra dispatch interval.")
        if case.batteries:
            for asset in case.batteries:
                energy = _series(result.batteries[asset.id], "energy_mwh", count+1)
                keep = np.unique(np.linspace(0, count, min(count+1, 512), dtype=int))
                axis.plot(edges[keep], energy[keep], linewidth=1.6, marker="o" if count <= 48 else None,
                          markersize=2.5, label=f"{asset.id} @ {asset.site} · capacity {asset.capacity_mwh:g} MWh")
            axis.legend(frameon=False, loc="best", fontsize=8)
            axis.set_ylim(bottom=0)
        else:
            axis.text(.5, .5, "Not applicable · no battery assets in this case", transform=axis.transAxes,
                      ha="center", va="center", fontsize=12, color="#64748b")
        axis.set_xlabel("Elapsed horizon time (h)")
        axis.set_ylabel("Stored energy (MWh)")
        axis.grid(axis="y")
        save(fig, "soc")
        fig, axis = plt.subplots(figsize=(11.2, 5.3))
        fig.subplots_adjust(left=.09, right=.97, bottom=.16, top=.82)
        header(fig, "supplied prices", "Prices follow the supplied interval inputs.")
        for site in case.sites:
            rows = case.periods[site.id]
            step(axis, [p.buy_price for p in rows], f"{site.id}: buy")
            step(axis, [p.sell_price for p in rows], f"{site.id}: sell", linestyle="--")
            if any(p.retail_price != 0 for p in rows):
                step(axis, [p.retail_price for p in rows], f"{site.id}: retail", linestyle=":")
        axis.set_xlabel("Elapsed horizon time (h)")
        axis.set_ylabel(f"Price ({currency}/MWh)")
        axis.grid(axis="y")
        axis.legend(ncol=2, frameon=False, fontsize=8, loc="best")
        save(fig, "prices")
        fig, axis = plt.subplots(figsize=(11.2, 5.3))
        fig.subplots_adjust(left=.09, right=.97, bottom=.16, top=.82)
        header(fig, "cumulative modeled cash", "Demand charges and fixed horizon costs are applied at the final boundary. Positive net cash is modeled profit.")
        cash, adjustment = _cash(case, result)
        times = np.r_[edges, edges[-1]]
        keep = np.unique(np.r_[np.linspace(0, len(cash)-2, min(len(cash)-1, 510), dtype=int),
                               np.argmin(cash), np.argmax(cash), len(cash)-1])
        axis.step(times[keep], cash[keep], where="post", color="#1479b8", linewidth=1.8, label="Dispatch")
        if baseline is not None and _get(baseline, "feasible", False):
            reference, _ = _cash(case, baseline)
            axis.step(times[keep], reference[keep], where="post", color="#8a98a9", linewidth=1.35, linestyle="--", label="Baseline")
        axis.axhline(0, color="#9ba9b8", linewidth=.7)
        axis.set_xlabel("Elapsed horizon time (h)")
        axis.set_ylabel(f"Cumulative modeled cash ({currency})")
        axis.grid(axis="y")
        axis.legend(frameon=False, loc="best")
        axis.annotate(f"{cash[-1]:,.4g} {currency}", (times[-1], cash[-1]), xytext=(-5, 7),
                      textcoords="offset points", ha="right", fontsize=10, fontweight="bold")
        axis.text(.02, .96, f"Final-boundary charges: {-adjustment:,.4g} {currency}", transform=axis.transAxes,
                  ha="left", va="top", fontsize=8, color="#64748b")
        save(fig, "cash")
    return files


def write_reports(case, result, directory, baseline=None, grid_check=None):
    """Write schedule/cash evidence only for feasible results; return summary dict."""
    case.validate()
    if result.feasible:
        result = _replayed(case, result, "Dispatch")
    if baseline is not None and _get(baseline, "feasible", False):
        from .management import baseline_case
        baseline = _replayed(baseline_case(case), baseline, "Baseline")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    protected = ("summary.json", "README.md", "topology.txt", "dispatch.csv", "battery_soc.csv", "generator_dispatch.csv",
                 "flexible_dispatch.csv", "cashflow.csv", "horizon_charges.csv", "INCOMPLETE")
    if any((directory/name).exists() for name in protected):
        raise ValueError("Report destination already contains economic report files; choose a new directory")
    feasible = bool(result.feasible)
    lines = result.accounting.get("lines", []) if feasible else []
    summary = {"schema":1, "role":case.role, "currency":case.currency, "status":result.status,
               "feasible":feasible, "optimal":bool(result.optimal),
               "objective_cost":result.objective_cost if feasible else None,
               "accounting":{k:v for k,v in result.accounting.items() if k != "lines"} if feasible else {},
               "cashflow_records":len(lines),
               "validation":{k:v for k,v in result.validation.items() if k != "accounting"},
               "solver":result.solver, "grid_check":grid_check,
               "summary_json":"summary.json", "artifacts":{"readme":"README.md", "topology_txt":"topology.txt"}}
    if baseline is not None:
        base = {"status":_get(baseline, "status"), "feasible":bool(_get(baseline, "feasible", False)),
                "optimal":bool(_get(baseline, "optimal", False))}
        if feasible and base["feasible"]:
            account = _get(baseline, "accounting", {})
            if account.get("currency", case.currency) != case.currency:
                raise ValueError("Cannot compare baseline costs in different currencies")
            base.update(total_cost=account["total_cost"], net_profit=account["net_profit"],
                        baseline_cost_minus_dispatch_cost=account["total_cost"]-result.accounting["total_cost"],
                        dispatch_profit_minus_baseline_profit=result.accounting["net_profit"]-account["net_profit"])
        summary["baseline"] = base
    (directory/"topology.txt").write_text(_topology(case), encoding="utf-8")
    if not feasible:
        (directory/"README.md").write_text(
            f"# Economic dispatch result\n\nStatus: **{result.status}**. The solver did not return a feasible schedule.\n\n"
            "`summary.json` contains solver status and validation findings. [Asset overview](topology.txt) records the input assets and meter limits.\n", encoding="utf-8")
        (directory/"summary.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return _plain(summary)
    (directory/"INCOMPLETE").write_text("Economic report export has not completed.\n", encoding="utf-8")
    count = len(next(iter(case.periods.values())))
    def dispatch_rows():
        for site in case.sites:
            values = result.site[site.id]
            data = {key:_series(values, key, count) for key in ("import_mw", "export_mw", "renewable_used_mw", "grid_mode")}
            for slot, period in enumerate(case.periods[site.id]):
                yield [site.id, slot, period.time, period.duration_h, period.buy_price, period.sell_price,
                       period.retail_price, period.load_mw, period.loss_mw, period.renewable_mw,
                       *[data[key][slot] for key in data]]
    _csv(directory/"dispatch.csv", "site,slot,time,duration_h,buy_price,sell_price,retail_price,load_mw,loss_mw,renewable_available_mw,import_mw,export_mw,renewable_used_mw,grid_mode".split(","), dispatch_rows())
    def battery_rows():
        for asset in case.batteries:
            values = result.batteries[asset.id]
            energy = _series(values, "energy_mwh", count+1)
            charge, discharge, mode = [_series(values, name, count) for name in ("charge_mw", "discharge_mw", "charge_mode")]
            rows = case.periods[asset.site]
            times = np.r_[0, np.cumsum([p.duration_h for p in rows])]
            for slot in range(count+1):
                yield [asset.id, asset.site, slot, rows[slot].time if slot < count else "END", times[slot],
                       energy[slot], charge[slot] if slot < count else "", discharge[slot] if slot < count else "",
                       mode[slot] if slot < count else "", asset.capacity_mwh, asset.soc_min_mwh,
                       asset.capacity_mwh if asset.soc_max_mwh is None else asset.soc_max_mwh]
    _csv(directory/"battery_soc.csv", "battery,site,boundary,time,elapsed_h,energy_mwh,charge_mw,discharge_mw,charge_mode,capacity_mwh,soc_min_mwh,soc_max_mwh".split(","), battery_rows())
    def generator_rows():
        for asset in case.generators:
            values = result.generators[asset.id]
            data = {key:_series(values, key, count) for key in ("p_mw", "on", "startup", "shutdown")}
            for slot, period in enumerate(case.periods[asset.site]):
                yield [asset.id, asset.site, slot, period.time, period.duration_h, *[data[key][slot] for key in data]]
    _csv(directory/"generator_dispatch.csv", "generator,site,slot,time,duration_h,p_mw,on,startup,shutdown".split(","), generator_rows())
    def flexible_rows():
        for asset in case.flexible_loads:
            power = _series(result.flexible[asset.id], "p_mw", count)
            for slot, period in enumerate(case.periods[asset.site]):
                yield [asset.id, asset.site, slot, period.time, period.duration_h, power[slot], power[slot]*period.duration_h]
    _csv(directory/"flexible_dispatch.csv", "flexible_load,site,slot,time,duration_h,p_mw,energy_mwh".split(","), flexible_rows())
    fields = list(dict.fromkeys(key for line in lines for key in line))
    _csv(directory/"cashflow.csv", fields or ["site", "slot", "time", "total_cost", "net_profit"],
         ([line.get(key, "") for key in fields] for line in lines))
    charge_rows = []
    for site in case.sites:
        peak = result.accounting["peak_import_mw"][site.id]
        charge_rows.append(["incremental_demand_charge", site.id,
                            (peak-site.historical_peak_mw)*site.demand_charge_per_mw, case.currency,
                            peak, site.historical_peak_mw, site.demand_charge_per_mw])
    charge_rows.append(["fixed_horizon_cost", "ALL", case.fixed_horizon_cost, case.currency, "", "", ""])
    _csv(directory/"horizon_charges.csv", "type,site,amount,currency,peak_import_mw,historical_peak_mw,rate_per_mw".split(","), charge_rows)
    summary["artifacts"].update(dispatch_csv="dispatch.csv", battery_csv="battery_soc.csv",
                                generator_csv="generator_dispatch.csv", flexible_csv="flexible_dispatch.csv",
                                cashflow_csv="cashflow.csv", horizon_charges_csv="horizon_charges.csv")
    summary["artifacts"].update(_plots(case, result, directory, baseline))
    state = "Optimal schedule" if result.optimal else "Feasible schedule; optimality not established"
    note = (f"# Economic dispatch result\n\n**{state}.** Solver status: `{result.status}`.\n\n"
            f"Role: `{case.role}`. Currency/unit label: `{case.currency}`. Full status, accounting, validation and any grid check are in `summary.json`.\n\n"
            "`dispatch.csv` records each site meter separately. `battery_soc.csv` has T+1 stored-energy boundary states; its final row is not an extra interval. Generator commitment and flexible-load tables retain all assets.\n\n"
            "`cashflow.csv` contains per-interval site costs and revenues. `horizon_charges.csv` contains incremental demand charges and fixed horizon costs; add these to interval total costs to reconcile the summary. Cumulative cash applies those charges at the final boundary.\n\n"
            "[Asset overview](topology.txt) shows each PCC and its attached assets.\n\n"
            "Plots are bounded overview previews; CSV files retain all intervals. Net profit means modeled horizon revenue minus the supplied costs. Prices and forecasts are supplied inputs. Net value covers the supplied horizon and cost assumptions.\n")
    if grid_check is None:
        note += "\nGrid-check status: not requested.\n"
    (directory/"README.md").write_text(note, encoding="utf-8")
    summary = _plain(summary)
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (directory/"INCOMPLETE").unlink()
    return summary
