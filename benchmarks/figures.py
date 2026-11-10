"""Bounded publication plots for solved NRlarge benchmark cases.

The network drawing groups conductor nodes at their physical bus and uses a
deterministic force layout. It does not infer geographic positions. Full numeric
exports remain the authoritative record; plots retain extrema when sampled.
"""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".task-check" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from PIL import Image
import networkx as nx
import numpy as np

from nrlarge.model import Network, evaluate


_INK = "#18283f"
_MUTED = "#5e6d80"
_GRID = "#e6ebf0"
_PHASES = {"A": "#1479b8", "B": "#df7b1c", "C": "#198b66",
           "S1": "#9058ac", "S2": "#d25376"}
_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": _INK,
    "axes.labelcolor": _INK, "axes.edgecolor": "#c5cfda",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": _MUTED, "ytick.color": _MUTED,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "grid.color": _GRID, "grid.linewidth": .7, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "path.simplify": True, "path.simplify_threshold": .15,
    "svg.hashsalt": "nrlarge-benchmark-gallery-v1",
}


def _bounded(indices, values, limit):
    """Evenly sample a stable ordering, retaining its finite minimum/maximum."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= limit:
        return indices
    take = np.linspace(0, len(indices)-1, max(0, limit-2), dtype=int)
    finite = np.flatnonzero(np.isfinite(np.asarray(values)[indices]))
    extremes = []
    if len(finite):
        subset = np.asarray(values)[indices[finite]]
        extremes = [finite[np.argmin(subset)], finite[np.argmax(subset)]]
    return indices[np.unique(np.r_[take, extremes]).astype(int)][:limit]


def _decorate(fig, title, subtitle, converged):
    fig.suptitle(title, x=.075, y=.975, ha="left", fontsize=17, fontweight="bold")
    fig.text(.075, .926, subtitle, ha="left", va="top", fontsize=9, color=_MUTED)
    if not converged:
        fig.text(.075, .887, "NOT CONVERGED · LAST ITERATE", ha="left", va="top",
                 fontsize=9, color="#b33a35", fontweight="bold")


def _save(fig, directory, stem, files):
    """Small fixed-resolution PNG plus text-preserving, zoomable SVG."""
    fig.savefig(directory / f"{stem}.svg", format="svg",
                metadata={"Date":None, "Creator":"NRlarge benchmark gallery"})
    raster = BytesIO()
    fig.savefig(raster, format="png", dpi=100)
    raster.seek(0)
    # Palette PNG previews keep the gallery small; the SVG retains exact colors
    # and vector geometry for publication/export at arbitrary resolution.
    with Image.open(raster) as image:
        preview = image.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT,
                                                 dither=Image.Dither.NONE)
        preview.save(directory / f"{stem}.png", optimize=True, compress_level=9)
    raster.close()
    plt.close(fig)
    files[f"{stem}_svg"] = f"{stem}.svg"
    files[f"{stem}_png"] = f"{stem}.png"


def _bus_ticks(axis, buses):
    ticks = np.unique(np.linspace(0, max(0, len(buses)-1), min(12, len(buses)), dtype=int))
    axis.set_xticks(ticks, [str(buses[k])[:22] for k in ticks], rotation=28, ha="right", fontsize=8)
    axis.set_xlabel("Physical bus (input order; not distance)")
    axis.set_xlim(-.6, max(.6, len(buses)-.4))


def _phase_profile(axis, case, values, bus_index, title, ylabel, point_limit=1000):
    present = [p for p in _PHASES if any(node.phase == p for node in case.nodes)]
    total = sum(node.phase != "N" for node in case.nodes)
    shown = 0
    limit = max(2, point_limit // max(1, len(present)))
    for phase in present:
        positions = np.array([i for i, node in enumerate(case.nodes) if node.phase == phase], dtype=int)
        selected = _bounded(positions, values, limit)
        if not len(selected):
            continue
        x = [bus_index[case.nodes[k].bus or case.nodes[k].id] for k in selected]
        axis.plot(x, np.asarray(values)[selected], color=_PHASES[phase], linewidth=.8,
                  marker="o", markevery=max(1, len(selected)//36),
                  markersize=3 if total <= 100 else 1.8, label=phase, alpha=.9)
        shown += len(selected)
    axis.set_title(title, loc="left", pad=12)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y")
    if present:
        axis.legend(title="Phase", ncol=len(present), frameon=False, loc="best", fontsize=8, title_fontsize=8)
    axis.text(1, 1.015, f"{shown:,} / {total:,} conductors" + (" · extrema retained" if shown < total else ""),
              transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color=_MUTED)


def _network(case, result, buses, node_bus, directory, title, files):
    bus_nodes = defaultdict(list)
    for i, node in enumerate(case.nodes):
        bus_nodes[node_bus[node.id]].append(i)
    edges = []
    current, pairs = None, set()
    # Case stamps are grouped contiguously by element (Network validates this).
    def finish():
        for a, b in sorted(pairs):
            edges.append((current, a, b))
    for element, row, col, value in case.stamps:
        if element != current:
            if current is not None:
                finish()
            current, pairs = element, set()
        a, b = node_bus[row], node_bus[col]
        if a != b and value != 0:
            pairs.add(tuple(sorted((a, b))))
    if current is not None:
        finish()
    graph = nx.Graph()
    graph.add_nodes_from(buses)
    graph.add_edges_from((a, b) for _, a, b in edges)
    displayed = buses
    if len(buses) > 1000:
        reference = [bus for bus in buses if any(case.nodes[k].type == "SLACK" for k in bus_nodes[bus])]
        sample = [buses[k] for k in np.linspace(0, len(buses)-1, 1000, dtype=int)]
        displayed = list(dict.fromkeys(reference + sample))[:1000]
        graph = graph.subgraph(displayed).copy()
    selected = set(displayed)
    visible_edges = [(e, a, b) for e, a, b in edges if a in selected and b in selected]
    if len(visible_edges) > 2000:
        visible_edges = [visible_edges[k] for k in np.linspace(0, len(visible_edges)-1, 2000, dtype=int)]
    if len(graph) == 1:
        layout = {displayed[0]: np.zeros(2)}
    else:
        layout = nx.spring_layout(graph, seed=27, iterations=100 if len(graph) <= 300 else 50,
                                  k=1.35 / np.sqrt(max(1, len(graph))), weight=None)
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.subplots_adjust(left=.045, right=.91, bottom=.11, top=.86)
    subtitle = f"{len(buses):,} physical buses · {len(case.nodes):,} conductors · {len(edges):,} element connections · {len(case.loads):,} explicit loads"
    _decorate(fig, title + " · network", subtitle, result.converged)
    groups = defaultdict(list)
    for element, a, b in visible_edges:
        groups[(a, b)].append(element)
    for (a, b), elements in groups.items():
        offsets = (np.arange(len(elements)) - (len(elements)-1)/2) * .17
        for element, curvature in zip(elements, offsets):
            arc = FancyArrowPatch(layout[a], layout[b], arrowstyle="-", color="#a9b5c4",
                                  linewidth=.8 if len(buses) < 100 else .55,
                                  connectionstyle=f"arc3,rad={curvature}", zorder=1,
                                  shrinkA=3, shrinkB=3)
            ax.add_patch(arc)
            if len(buses) <= 16:
                center = (np.asarray(layout[a])+np.asarray(layout[b]))/2
                difference = np.asarray(layout[b])-np.asarray(layout[a])
                center += np.array([-difference[1], difference[0]]) * curvature * .5
                ax.text(*center, str(element)[:22], ha="center", va="center", fontsize=6.3,
                        color="#6d7a8a", zorder=2,
                        bbox={"facecolor":"white", "edgecolor":"none", "alpha":.8, "pad":.3})
    vm = np.abs(result.voltage)
    colors = {}
    for bus in displayed:
        phase = [k for k in bus_nodes[bus] if case.nodes[k].phase != "N"]
        colors[bus] = float(min(vm[phase])) if phase else 0.0
    low, high = min(colors.values()), max(colors.values())
    if high-low < 1e-9:
        low, high = low-.005, high+.005
    normalizer = matplotlib.colors.Normalize(low, high)
    cmap = plt.get_cmap("viridis")
    kinds = {"SLACK": [], "PV": [], "PQ": []}
    for bus in displayed:
        types = {case.nodes[k].type for k in bus_nodes[bus]}
        kinds["SLACK" if "SLACK" in types else "PV" if "PV" in types else "PQ"].append(bus)
    base = 100 if len(buses) <= 30 else 40 if len(buses) <= 120 else 24
    for kind, marker in (("PQ", "o"), ("PV", "D"), ("SLACK", "s")):
        subset = kinds[kind]
        if subset:
            ax.scatter([layout[b][0] for b in subset], [layout[b][1] for b in subset],
                       c=[colors[b] for b in subset], norm=normalizer, cmap=cmap,
                       s=base*(1.65 if kind != "PQ" else 1), marker=marker,
                       edgecolors="#26364b" if kind != "PQ" else "white",
                       linewidths=.65 if kind != "PQ" else .3, zorder=3)
    if len(buses) <= 25:
        labels = displayed
    else:
        important = list(dict.fromkeys(kinds["SLACK"] + kinds["PV"]))
        by_degree = sorted(displayed, key=lambda bus: (-graph.degree(bus), str(bus)))
        labels = list(dict.fromkeys(important + by_degree))[:24 if len(buses) <= 120 else 18]
    for bus in labels:
        ax.annotate(str(bus)[:25], layout[bus], xytext=(4, 4), textcoords="offset points",
                    fontsize=8 if len(buses) <= 25 else 6.5, color=_INK,
                    bbox={"facecolor":"white", "edgecolor":"none", "alpha":.75, "pad":.6}, zorder=4)
    ax.set_axis_off()
    ax.margins(.12)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=ax,
                       fraction=.025, pad=.025, shrink=.7)
    bar.set_label("Minimum phase |V| at physical bus (pu)", fontsize=9)
    handles = [Line2D([], [], marker=marker, linestyle="", color=_INK, markersize=6, label=label)
               for marker, label in (("s", "Slack / reference"), ("D", "PV / generator"), ("o", "PQ / load bus"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.48, .045),
               ncol=3, frameon=False, fontsize=9)
    message = "Deterministic topology layout · not geographic · parallel elements use separate arcs"
    if len(displayed) < len(buses) or len(visible_edges) < len(edges):
        message += f"\nSampled view: {len(displayed):,} buses and {len(visible_edges):,} connections; full data remain in CSV."
    fig.text(.075, .016, message, fontsize=8, color=_MUTED)
    _save(fig, directory, "network", files)


def render_case(case, result, output_dir, title):
    """Write five SVG/PNG plot pairs; return relative filenames keyed by plot/type.

    Charts show at most 1000 points, retaining voltage/angle extrema. The network
    view retains every branch for the 118/300-bus gallery cases; larger views are
    bounded to 1000 physical buses and 2000 element connections and marked sampled.
    Result powers are derived from actual solved voltages, including ZIP demand.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if len(result.voltage) != len(case.nodes):
        raise ValueError("Result voltage count does not match case conductor count")
    net = Network(case)
    _, actual, load_i = evaluate(net, result.voltage, result.types, jacobian=False)
    node_bus = {node.id: node.bus or node.id for node in case.nodes}
    buses = list(dict.fromkeys(node_bus.values()))
    bus_index = {bus: k for k, bus in enumerate(buses)}
    bus_groups = defaultdict(list)
    for k, node in enumerate(case.nodes):
        bus_groups[node_bus[node.id]].append(k)
    vm, angles = np.abs(result.voltage), np.angle(result.voltage, deg=True)
    files = {}
    subtitle = f"{len(buses):,} physical buses · {len(case.nodes):,} conductors · {len(case.loads):,} explicit loads · three-phase base {case.base_mva:g} MVA"
    with matplotlib.rc_context(_STYLE):
        _network(case, result, buses, node_bus, directory, title, files)
        neutrals = np.array([k for k, node in enumerate(case.nodes) if node.phase == "N"], dtype=int)
        wye = [k for k, load in enumerate(case.loads)
               if load.target == "GROUND" or case.nodes[net.index[load.target]].phase == "N"]
        panels = 1 + bool(len(neutrals)) + bool(len(neutrals) and wye)
        fig, axes = plt.subplots(panels, 1, figsize=(11.5, 7.5 if panels > 1 else 6.4), squeeze=False)
        fig.subplots_adjust(left=.1, right=.97, bottom=.14, top=.83, hspace=.85)
        _decorate(fig, title + " · voltage magnitudes", subtitle, result.converged)
        panel_budget = 1000 // panels
        _phase_profile(axes[0, 0], case, vm, bus_index, "Phase conductors to reference", "Voltage magnitude (pu)", panel_budget)
        _bus_ticks(axes[0, 0], buses)
        if len(neutrals):
            selected = _bounded(neutrals, vm, panel_budget)
            axis = axes[1, 0]
            axis.plot([bus_index[node_bus[case.nodes[k].id]] for k in selected], vm[selected],
                      color="#a24e8a", marker="o", markersize=3, linewidth=.8)
            axis.set_title("Neutral displacement to reference", loc="left", pad=11)
            axis.set_ylabel("|V_N| (pu)")
            axis.grid(axis="y")
            _bus_ticks(axis, buses)
            if wye:
                terminal = np.abs(net.a @ result.voltage) / net.nominal
                selected_load = _bounded(wye, terminal, panel_budget)
                axis = axes[2, 0]
                axis.plot(np.asarray(selected_load)+1, terminal[selected_load], color="#176f8a",
                          marker="o", markersize=3, linewidth=.8)
                axis.set_title("Wye customer terminal voltage · phase to its actual neutral / ground", loc="left", pad=11)
                axis.set_ylabel("Terminal / nominal (pu)")
                axis.set_xlabel("Wye load record number (input order)")
                axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                axis.grid(axis="y")
        fig.text(.1, .018, "Neutral conductors are excluded from phase profiles. Voltage values are not automatically checked against operating limits.",
                 fontsize=8, color=_MUTED)
        _save(fig, directory, "voltage", files)
        fig, axis = plt.subplots(figsize=(11.5, 6.4))
        fig.subplots_adjust(left=.1, right=.97, bottom=.18, top=.82)
        _decorate(fig, title + " · voltage angles", subtitle, result.converged)
        _phase_profile(axis, case, angles, bus_index, "Phase angles relative to case reference", "Voltage angle (degrees)")
        _bus_ticks(axis, buses)
        fig.text(.1, .018, "Phase A/B/C may be centered near 0 / −120 / +120 degrees; neutral angles are omitted.", fontsize=8, color=_MUTED)
        _save(fig, directory, "angles", files)
        fig, axis = plt.subplots(figsize=(11.5, 6.4))
        fig.subplots_adjust(left=.1, right=.97, bottom=.15, top=.82)
        _decorate(fig, title + " · Newton convergence", subtitle, result.converged)
        floor = 1e-16
        for key, label, color in (("residual", "Mixed-equation residual", "#1479b8"),
                                  ("power_error", "PQ complex-power error", "#df7b1c")):
            values = np.array([row.get(key, np.nan) for row in result.history], dtype=float)
            indices = _bounded(np.flatnonzero(np.isfinite(values)), values, 500)
            axis.plot([result.history[k].get("iteration", k) for k in indices], np.maximum(values[indices], floor),
                      color=color, marker="o", markersize=4, linewidth=1.5, label=label)
        axis.set_yscale("log")
        axis.set_xlabel("Newton update number (0 = initial state)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axis.set_ylabel("Maximum error (per unit; equation scaling differs)")
        axis.grid(which="major", axis="both")
        axis.legend(frameon=False, loc="lower left")
        final = result.history[-1] if result.history else {}
        receipt = f"Status: {result.reason}\nUpdates: {max(0, len(result.history)-1)}\nSolve time: {result.seconds:.4g} s\nFinal residual: {final.get('residual', float('nan')):.3g}"
        axis.text(.98, .96, receipt, transform=axis.transAxes, ha="right", va="top", fontsize=9,
                  bbox={"boxstyle":"round,pad=.7", "facecolor":"#f4f7fa", "edgecolor":_GRID})
        fig.text(.1, .018, "Zero errors are drawn at 1e−16. Ordinary rows use current balance; PV rows use P balance and squared voltage.", fontsize=8, color=_MUTED)
        _save(fig, directory, "convergence", files)
        load_power = (net.a @ result.voltage) * np.conj(load_i)
        absorbed = np.sum(result.voltage * np.conj(net.y @ result.voltage))
        scale = case.base_mva / 3
        # Zero-injection PQ conductors are not extra generators. Counting their
        # unconverged residuals as solved sources would force an algebraic balance
        # identity and hide the failure. Preserve the original source contract.
        source_mask = np.array([node.type in ("SLACK", "PV") or node.p != 0 or node.q != 0
                                for node in case.nodes], dtype=bool)
        sources = actual[source_mask]
        generation = (np.maximum(sources.real, 0).sum() + np.maximum(-load_power.real, 0).sum()) * scale
        demand = (np.maximum(-sources.real, 0).sum() + np.maximum(load_power.real, 0).sum()) * scale
        active = np.array([generation, demand, absorbed.real*scale])
        reactive = np.array([sources.imag.sum(), load_power.imag.sum(), absorbed.imag]) * scale
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4))
        fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.8, wspace=.35)
        balance_title = "power balance" if result.converged else "last-iterate power balance"
        _decorate(fig, title + " · " + balance_title, subtitle, result.converged)
        for axis, values, unit, labels in (
                (axes[0], active, "Active power (MW)", ["Generation", "Demand", "Element net\nabsorption"]),
                (axes[1], reactive, "Reactive power (MVAr)", ["Net nodal\ninjection", "Explicit ZIP\ndemand", "Element net\nabsorption"])):
            axis.bar(range(3), values, color=["#1479b8", "#df7b1c", "#198b66"], width=.6)
            axis.axhline(0, color="#94a3b5", linewidth=.8)
            axis.set_xticks(range(3), labels, fontsize=9)
            axis.set_ylabel(unit)
            axis.grid(axis="y")
            axis.set_axisbelow(True)
            span = max(float(np.ptp(np.r_[values, 0])), 1e-9)
            axis.set_ylim(min(0, values.min())-.17*span, max(0, values.max())+.2*span)
            for k, value in enumerate(values):
                axis.annotate(f"{value:,.5g}", (k, value), xytext=(0, 5 if value >= 0 else -6),
                              textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                              fontsize=9, fontweight="bold")
        active_title = ("Generation − demand = element absorption" if result.converged
                        else "Generation, demand and element absorption")
        axes[0].set_title(active_title, loc="left", fontsize=11, pad=12)
        axes[1].set_title("Signed reactive-power balance", loc="left", fontsize=11, pad=12)
        p_error = active[0]-active[1]-active[2]
        q_error = reactive[0]-reactive[1]-reactive[2]
        fig.text(.09, .135, f"Balance closure: ΔP = {p_error:.3g} MW · ΔQ = {q_error:.3g} MVAr", fontsize=10, color=_INK)
        fig.text(.09, .055, "Demand includes negative specified-node injections and actual voltage-dependent ZIP load consumption.\nOnly original SLACK/PV or specified-injection conductors count as nodal sources; zero-injection residuals remain in the closure gap.\nElement absorption includes lines, transformers and shunts; reactive absorption may be negative due to charging.",
                 fontsize=8, color=_MUTED, linespacing=1.6)
        _save(fig, directory, "power", files)
    return files
