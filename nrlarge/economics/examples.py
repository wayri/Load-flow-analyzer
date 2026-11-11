"""Editable synthetic 24-hour cases; no live prices, forecasts or tariffs.

Prices and forecasts are synthetic inputs in currency units per MWh and MW.
Net value covers the supplied horizon and cost assumptions.
"""
import math

from .model import Battery, Case, FlexibleLoad, Generator, Interval, Site


def _prices(hour):
    buy = 32.0 if hour < 6 else 58.0 if hour < 16 else 158.0 if hour < 22 else 45.0
    return buy, buy * .72


def _solar(hour, maximum):
    return maximum * max(0.0, math.sin(math.pi * (hour - 6) / 12))


def _periods(load=None, renewable=None, retail=0.0, loss_fraction=0.0, price_multiplier=1.0):
    rows = []
    for hour in range(24):
        buy, sell = _prices(hour)
        demand = 0.0 if load is None else float(load(hour))
        rows.append(Interval(time=f"2030-01-15T{hour:02d}:00:00+00:00", duration_h=1.0,
                             buy_price=buy*price_multiplier, sell_price=sell*price_multiplier,
                             load_mw=demand,
                             renewable_mw=0.0 if renewable is None else float(renewable(hour)),
                             retail_price=retail, loss_mw=demand*loss_fraction))
    return rows


def example_case(role):
    """Return a complete battery, generator, DISCOM or two-meter aggregator case.

    Every battery has a terminal SOC equal to its initial SOC, so the optimizer
    cannot claim revenue by consuming an unreplenished starting energy inventory.
    Each aggregator site has its own meter limits and balance equation.
    """
    if role == "battery":
        return Case(
            sites=[Site("battery-meter", import_limit_mw=2.0, export_limit_mw=2.0)],
            periods={"battery-meter": _periods()},
            batteries=[Battery("B1", "battery-meter", capacity_mwh=6.0,
                               charge_max_mw=2.0, discharge_max_mw=2.0,
                               initial_soc_mwh=3.0, terminal_soc_mwh=3.0,
                               soc_min_mwh=.6, soc_max_mwh=5.4,
                               degradation_cost_per_mwh=2.0, max_discharge_mwh=9.0)],
            currency="CUR", role="battery", fixed_horizon_cost=10.0)
    if role == "generator":
        return Case(
            sites=[Site("generator-meter", import_limit_mw=1.0, export_limit_mw=4.0)],
            periods={"generator-meter": _periods()},
            generators=[Generator("G1", "generator-meter", p_max_mw=4.0,
                                  marginal_cost_per_mwh=67.0, p_min_mw=.8,
                                  no_load_cost_per_h=6.0, startup_cost=80.0,
                                  ramp_up_mw_per_h=1.5, ramp_down_mw_per_h=1.5,
                                  startup_ramp_mw=1.5, shutdown_ramp_mw=1.5,
                                  min_up_h=3.0, min_down_h=2.0, initial_state_h=4.0)],
            currency="CUR", role="generator", fixed_horizon_cost=20.0)
    if role == "discom":
        def demand(hour):
            return 4.6 + (1.6 if 17 <= hour < 22 else .5 if 7 <= hour < 17 else 0.0)
        return Case(
            sites=[Site("distribution-meter", import_limit_mw=8.0, export_limit_mw=2.0,
                        demand_charge_per_mw=6.0, historical_peak_mw=4.0)],
            periods={"distribution-meter": _periods(demand, lambda hour: _solar(hour, 2.2),
                                                       retail=112.0, loss_fraction=.03)},
            batteries=[Battery("substation-B1", "distribution-meter", capacity_mwh=8.0,
                               charge_max_mw=2.0, discharge_max_mw=2.0,
                               initial_soc_mwh=2.0, terminal_soc_mwh=2.0,
                               soc_min_mwh=.8, soc_max_mwh=7.2,
                               degradation_cost_per_mwh=1.5)],
            flexible_loads=[FlexibleLoad("overnight-EV", "distribution-meter", energy_mwh=3.0,
                                         p_max_mw=1.5, start_slot=0, end_slot=6,
                                         value_per_mwh=112.0)],
            currency="CUR", role="discom", fixed_horizon_cost=75.0,
            renewable_curtail_cost_per_mwh=.5)
    if role == "aggregator":
        return Case(
            sites=[Site("depot-meter", import_limit_mw=2.0, export_limit_mw=1.5,
                        demand_charge_per_mw=3.0, historical_peak_mw=.5),
                   Site("roof-meter", import_limit_mw=1.5, export_limit_mw=1.2,
                        demand_charge_per_mw=2.0, historical_peak_mw=.3)],
            periods={"depot-meter": _periods(lambda hour: .35 + (.35 if 8 <= hour < 18 else 0.0),
                                              lambda hour: _solar(hour, .8), retail=95.0),
                     "roof-meter": _periods(lambda hour: .25,
                                             lambda hour: _solar(hour, 2.0), retail=95.0,
                                             price_multiplier=1.08)},
            batteries=[Battery("depot-B1", "depot-meter", capacity_mwh=3.0,
                               charge_max_mw=1.0, discharge_max_mw=1.0,
                               initial_soc_mwh=1.0, terminal_soc_mwh=1.0,
                               degradation_cost_per_mwh=2.0),
                       Battery("roof-B1", "roof-meter", capacity_mwh=4.0,
                               charge_max_mw=1.2, discharge_max_mw=1.2,
                               initial_soc_mwh=1.5, terminal_soc_mwh=1.5,
                               soc_min_mwh=.4, soc_max_mwh=3.6,
                               degradation_cost_per_mwh=2.0)],
            flexible_loads=[FlexibleLoad("depot-EV", "depot-meter", energy_mwh=4.0,
                                         p_max_mw=1.5, start_slot=7, end_slot=18,
                                         value_per_mwh=100.0),
                            FlexibleLoad("roof-process", "roof-meter", energy_mwh=2.0,
                                         p_max_mw=.8, start_slot=9, end_slot=16,
                                         value_per_mwh=100.0)],
            currency="CUR", role="aggregator", fixed_horizon_cost=25.0,
            renewable_curtail_cost_per_mwh=1.0)
    raise ValueError("Role must be battery, generator, discom or aggregator")
