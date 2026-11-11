"""Dispatch inputs. Power MW, energy MWh, prices currency/MWh, durations hours."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math


@dataclass
class Site:
    id: str
    import_limit_mw: float
    export_limit_mw: float
    demand_charge_per_mw: float = 0
    historical_peak_mw: float = 0


@dataclass
class Interval:
    time: str
    duration_h: float
    buy_price: float
    sell_price: float
    load_mw: float = 0
    renewable_mw: float = 0
    retail_price: float = 0
    loss_mw: float = 0


@dataclass
class Battery:
    id: str
    site: str
    capacity_mwh: float
    charge_max_mw: float
    discharge_max_mw: float
    eta_charge: float = .95
    eta_discharge: float = .95
    soc_min_mwh: float = 0
    soc_max_mwh: float | None = None
    initial_soc_mwh: float = 0
    terminal_soc_mwh: float | None = None
    degradation_cost_per_mwh: float = 0
    self_discharge_per_h: float = 0
    max_discharge_mwh: float | None = None


@dataclass
class Generator:
    id: str
    site: str
    p_max_mw: float
    marginal_cost_per_mwh: float
    p_min_mw: float = 0
    no_load_cost_per_h: float = 0
    startup_cost: float = 0
    ramp_up_mw_per_h: float | None = None
    ramp_down_mw_per_h: float | None = None
    startup_ramp_mw: float | None = None
    shutdown_ramp_mw: float | None = None
    initial_on: int = 0
    initial_p_mw: float = 0
    min_up_h: float = 0
    min_down_h: float = 0
    initial_state_h: float = 0


@dataclass
class FlexibleLoad:
    id: str
    site: str
    energy_mwh: float
    p_max_mw: float
    start_slot: int = 0
    end_slot: int | None = None
    value_per_mwh: float = 0


def instant(value):
    if not isinstance(value,str):
        raise ValueError('Interval time must be an ISO8601 string with UTC offset')
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Interval times require an explicit UTC offset')
    return value.astimezone(timezone.utc)


@dataclass
class Case:
    sites: list[Site]
    periods: dict[str, list[Interval]]
    batteries: list[Battery] = field(default_factory=list)
    generators: list[Generator] = field(default_factory=list)
    flexible_loads: list[FlexibleLoad] = field(default_factory=list)
    currency: str = 'CUR'
    role: str = 'battery'
    renewable_curtail_cost_per_mwh: float = 0
    fixed_horizon_cost: float = 0

    def validate(self):
        def number(x, name, minimum=None):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f'{name} must be finite numeric data')
            if minimum is not None and x < minimum:
                raise ValueError(f'{name} must be >= {minimum}')
        def unique(items, label):
            ids = [x.id for x in items]
            if any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
                raise ValueError(f'{label} IDs must be unique nonempty strings')
        unique(self.sites, 'site')
        if not self.sites or set(self.periods) != {s.id for s in self.sites}:
            raise ValueError('Every site requires its own periods')
        reference = None
        for s in self.sites:
            for name in ('import_limit_mw', 'export_limit_mw', 'demand_charge_per_mw', 'historical_peak_mw'):
                number(getattr(s, name), name, 0)
            rows = self.periods[s.id]
            if not rows or len(rows) > 87600:
                raise ValueError('Use 1..87600 intervals per site')
            times = []
            for r in rows:
                for name in ('duration_h', 'buy_price', 'sell_price', 'load_mw', 'renewable_mw', 'retail_price', 'loss_mw'):
                    number(getattr(r, name), name, 0 if name in ('load_mw', 'renewable_mw', 'loss_mw') else None)
                if r.duration_h <= 0:
                    raise ValueError('duration_h must be positive')
                times.append((instant(r.time), r.duration_h))
            for (a, dt), (b, _) in zip(times, times[1:]):
                if abs((b-a).total_seconds()/3600-dt) > 1e-8:
                    raise ValueError('Intervals must be ordered and contiguous')
            if reference is not None and times != reference:
                raise ValueError('All sites must share interval boundaries')
            reference = times
        n = len(reference)
        ids = {s.id for s in self.sites}
        for items, label in ((self.batteries,'battery'),(self.generators,'generator'),(self.flexible_loads,'flexible load')):
            unique(items,label)
            for a in items:
                if a.site not in ids:
                    raise ValueError(f'Unknown site {a.site}')
        for b in self.batteries:
            for name in ('capacity_mwh','charge_max_mw','discharge_max_mw','soc_min_mwh','initial_soc_mwh',
                         'degradation_cost_per_mwh','self_discharge_per_h'):
                number(getattr(b,name),name,0)
            for name in ('soc_max_mwh','terminal_soc_mwh','max_discharge_mwh'):
                if getattr(b,name) is not None:
                    number(getattr(b,name),name,0)
            for name in ('eta_charge','eta_discharge'):
                number(getattr(b,name),name)
                if not 0 < getattr(b,name) <= 1:
                    raise ValueError('Battery efficiencies must be in (0,1]')
            hi = b.capacity_mwh if b.soc_max_mwh is None else b.soc_max_mwh
            end = b.initial_soc_mwh if b.terminal_soc_mwh is None else b.terminal_soc_mwh
            if not (0 <= b.soc_min_mwh <= b.initial_soc_mwh <= hi <= b.capacity_mwh and b.soc_min_mwh <= end <= hi):
                raise ValueError('Battery SOC bounds/initial/terminal must fit capacity')
            if b.self_discharge_per_h >= 1:
                raise ValueError('Self-discharge must be in [0,1)')
        for g in self.generators:
            for name in ('p_max_mw','p_min_mw','no_load_cost_per_h','startup_cost','initial_p_mw','min_up_h','min_down_h','initial_state_h'):
                number(getattr(g,name),name,0)
            number(g.marginal_cost_per_mwh,'marginal_cost_per_mwh')
            for name in ('ramp_up_mw_per_h','ramp_down_mw_per_h','startup_ramp_mw','shutdown_ramp_mw'):
                if getattr(g,name) is not None:
                    number(getattr(g,name),name,0)
            if g.initial_on not in (0,1) or g.p_min_mw > g.p_max_mw:
                raise ValueError('Invalid generator initial state or output bounds')
            if not g.p_min_mw*g.initial_on <= g.initial_p_mw <= g.p_max_mw*g.initial_on:
                raise ValueError('Initial generator output must match commitment')
        for f in self.flexible_loads:
            number(f.energy_mwh,'energy_mwh',0); number(f.p_max_mw,'p_max_mw',0)
            number(f.value_per_mwh,'value_per_mwh')
            end = n if f.end_slot is None else f.end_slot
            if type(f.start_slot) is not int or type(end) is not int or not 0 <= f.start_slot <= end <= n:
                raise ValueError('Flexible windows use integer [start_slot,end_slot) within horizon')
        number(self.renewable_curtail_cost_per_mwh,'renewable_curtail_cost_per_mwh',0)
        number(self.fixed_horizon_cost,'fixed_horizon_cost',0)
        if not isinstance(self.currency,str) or not self.currency.strip():
            raise ValueError('A currency/unit label is required')
        return self
