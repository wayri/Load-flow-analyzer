"""Sparse mixed-integer dispatch model solved by SciPy/HiGHS."""
from dataclasses import dataclass, field
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class Dispatch:
    status: str
    optimal: bool = False
    feasible: bool = False
    site: dict = field(default_factory=dict)
    batteries: dict = field(default_factory=dict)
    generators: dict = field(default_factory=dict)
    flexible: dict = field(default_factory=dict)
    accounting: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)
    objective_cost: float | None = None


class _MILP:
    def __init__(self):
        self.c=[]; self.lo=[]; self.hi=[]; self.integer=[]
        self.rr=[]; self.cc=[]; self.vv=[]; self.lb=[]; self.ub=[]

    def var(self, n, low=0, high=math.inf, cost=0, integer=False):
        if len(self.c)+n > 100000:
            raise ValueError('Dispatch model exceeds 100000-variable budget; shorten horizon or aggregate assets')
        start=len(self.c)
        for target,value in ((self.c,cost),(self.lo,low),(self.hi,high),(self.integer,int(integer))):
            target.extend(np.broadcast_to(value,(n,)).tolist())
        return np.arange(start,start+n)

    def row(self, terms, low=-math.inf, high=math.inf):
        terms={int(k):float(v) for k,v in terms.items() if v}
        if len(self.lb)>=500000 or len(self.vv)+len(terms)>2000000:
            raise ValueError('Dispatch constraint budget exceeded')
        i=len(self.lb)
        for k,v in terms.items():
            self.rr.append(i); self.cc.append(k); self.vv.append(v)
        self.lb.append(low); self.ub.append(high)

    def run(self, seconds, gap):
        a=coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.lb),len(self.c))).tocsc()
        return milp(np.array(self.c),integrality=np.array(self.integer),
                    bounds=Bounds(self.lo,self.hi),constraints=LinearConstraint(a,self.lb,self.ub),
                    options={'time_limit':seconds,'mip_rel_gap':gap})


def optimize(case, time_limit=60, mip_rel_gap=1e-8):
    case.validate()
    if not math.isfinite(time_limit) or time_limit<=0 or not math.isfinite(mip_rel_gap) or mip_rel_gap<0:
        raise ValueError('Positive finite time_limit and nonnegative mip_rel_gap required')
    started=time.perf_counter(); m=_MILP()
    n=len(next(iter(case.periods.values())))
    dt=np.array([r.duration_h for r in next(iter(case.periods.values()))])
    starts=np.r_[0,np.cumsum(dt)[:-1]]
    sites={}; batteries={}; generators={}; flexible={}
    for s in case.sites:
        rows=case.periods[s.id]
        v={'import_mw':m.var(n,high=s.import_limit_mw,cost=dt*np.array([r.buy_price for r in rows])),
           'export_mw':m.var(n,high=s.export_limit_mw,cost=-dt*np.array([r.sell_price for r in rows])),
           'renewable_used_mw':m.var(n,high=[r.renewable_mw for r in rows],cost=-dt*case.renewable_curtail_cost_per_mwh),
           'grid_mode':m.var(n,high=1,integer=True),
           'peak':m.var(1,low=s.historical_peak_mw,high=max(s.historical_peak_mw,s.import_limit_mw),cost=s.demand_charge_per_mw)}
        sites[s.id]=v
        for t in range(n):
            m.row({v['import_mw'][t]:1,v['grid_mode'][t]:-s.import_limit_mw},high=0)
            m.row({v['export_mw'][t]:1,v['grid_mode'][t]:s.export_limit_mw},high=s.export_limit_mw)
            m.row({v['import_mw'][t]:1,v['peak'][0]:-1},high=0)
    for b in case.batteries:
        upper=b.capacity_mwh if b.soc_max_mwh is None else b.soc_max_mwh
        v={'charge_mw':m.var(n,high=b.charge_max_mw),
           'discharge_mw':m.var(n,high=b.discharge_max_mw,cost=dt*b.degradation_cost_per_mwh),
           'energy_mwh':m.var(n+1,low=b.soc_min_mwh,high=upper),
           'charge_mode':m.var(n,high=1,integer=True)}
        batteries[b.id]=v
        m.row({v['energy_mwh'][0]:1},b.initial_soc_mwh,b.initial_soc_mwh)
        end=b.initial_soc_mwh if b.terminal_soc_mwh is None else b.terminal_soc_mwh
        m.row({v['energy_mwh'][-1]:1},end,end)
        for t in range(n):
            m.row({v['charge_mw'][t]:1,v['charge_mode'][t]:-b.charge_max_mw},high=0)
            m.row({v['discharge_mw'][t]:1,v['charge_mode'][t]:b.discharge_max_mw},high=b.discharge_max_mw)
            m.row({v['energy_mwh'][t+1]:1,v['energy_mwh'][t]:-(1-b.self_discharge_per_h)**dt[t],
                   v['charge_mw'][t]:-b.eta_charge*dt[t],v['discharge_mw'][t]:dt[t]/b.eta_discharge},0,0)
        if b.max_discharge_mwh is not None:
            m.row(dict(zip(v['discharge_mw'],dt)),high=b.max_discharge_mwh)
    for g in case.generators:
        v={'p_mw':m.var(n,high=g.p_max_mw,cost=dt*g.marginal_cost_per_mwh),
           'on':m.var(n,high=1,cost=dt*g.no_load_cost_per_h,integer=True),
           'startup':m.var(n,high=1,cost=g.startup_cost,integer=True),
           'shutdown':m.var(n,high=1,integer=True)}
        generators[g.id]=v
        for t in range(n):
            p,u,st,sd=[v[k][t] for k in ('p_mw','on','startup','shutdown')]
            m.row({p:1,u:-g.p_max_mw},high=0)
            m.row({p:1,u:-g.p_min_mw},low=0)
            transition={u:1,st:-1,sd:1}
            if t: transition[v['on'][t-1]]=-1
            previous_on=g.initial_on if not t else 0
            m.row(transition,previous_on,previous_on)
            m.row({st:1,sd:1},high=1)
            ru=g.p_max_mw if g.ramp_up_mw_per_h is None else g.ramp_up_mw_per_h*dt[t]
            rd=g.p_max_mw if g.ramp_down_mw_per_h is None else g.ramp_down_mw_per_h*dt[t]
            su=g.p_max_mw if g.startup_ramp_mw is None else g.startup_ramp_mw
            sh=g.p_max_mw if g.shutdown_ramp_mw is None else g.shutdown_ramp_mw
            up={p:1,st:-su}; down={p:-1,u:-rd,sd:-sh}
            if t:
                up[v['p_mw'][t-1]]=-1; up[v['on'][t-1]]=-ru
                down[v['p_mw'][t-1]]=1
                m.row(up,high=0); m.row(down,high=0)
            else:
                m.row(up,high=g.initial_p_mw+ru*g.initial_on)
                m.row(down,high=-g.initial_p_mw)
            for k in range(t,n):
                elapsed=starts[k]-starts[t]
                if elapsed+1e-9<g.min_up_h: m.row({v['on'][k]:1,st:-1},low=0)
                if elapsed+1e-9<g.min_down_h: m.row({v['on'][k]:1,sd:1},high=1)
                if elapsed+1e-9>=max(g.min_up_h,g.min_down_h): break
            remaining=(g.min_up_h if g.initial_on else g.min_down_h)-g.initial_state_h
            if starts[t]+1e-9<remaining:
                m.row({u:1},g.initial_on,g.initial_on)
    for f in case.flexible_loads:
        end=n if f.end_slot is None else f.end_slot
        upper=[f.p_max_mw if f.start_slot<=t<end else 0 for t in range(n)]
        v={'p_mw':m.var(n,high=upper,cost=-dt*f.value_per_mwh)}
        flexible[f.id]=v
        m.row(dict(zip(v['p_mw'],dt)),f.energy_mwh,f.energy_mwh)
    for s in case.sites:
        v=sites[s.id]
        for t,r in enumerate(case.periods[s.id]):
            terms={v['import_mw'][t]:1,v['export_mw'][t]:-1,v['renewable_used_mw'][t]:1}
            for b in case.batteries:
                if b.site==s.id:
                    terms[batteries[b.id]['discharge_mw'][t]]=1
                    terms[batteries[b.id]['charge_mw'][t]]=-1
            for g in case.generators:
                if g.site==s.id: terms[generators[g.id]['p_mw'][t]]=1
            for f in case.flexible_loads:
                if f.site==s.id: terms[flexible[f.id]['p_mw'][t]]=-1
            m.row(terms,r.load_mw+r.loss_mw,r.load_mw+r.loss_mw)
    raw=m.run(time_limit,mip_rel_gap)
    status={0:'optimal',1:'feasible_limit' if raw.x is not None else 'limit_no_solution',2:'infeasible',3:'unbounded',4:'error'}.get(raw.status,'error')
    result=Dispatch(status=status,solver={'status':int(raw.status),'message':str(raw.message),
        'seconds':time.perf_counter()-started,'variables':len(m.c),'constraints':len(m.lb),
        'mip_gap':getattr(raw,'mip_gap',None),'mip_node_count':getattr(raw,'mip_node_count',None),
        'mip_dual_bound_variable_cost':getattr(raw,'mip_dual_bound',None)})
    if raw.x is None or raw.status not in (0,1): return result
    def unpack(mapping):
        return {key:{name:np.asarray(raw.x[idx],dtype=float).tolist() for name,idx in values.items() if name!='peak'}
                for key,values in mapping.items()}
    result.site=unpack(sites); result.batteries=unpack(batteries)
    result.generators=unpack(generators); result.flexible=unpack(flexible)
    from .validation import validate_dispatch
    result.validation=validate_dispatch(case,result)
    result.accounting=result.validation['accounting']
    constant=case.fixed_horizon_cost
    for s in case.sites:
        constant-=s.demand_charge_per_mw*s.historical_peak_mw
        constant+=sum((r.renewable_mw*case.renewable_curtail_cost_per_mwh-r.load_mw*r.retail_price)*r.duration_h for r in case.periods[s.id])
    expected=float(raw.fun)+constant
    if abs(expected-result.accounting['total_cost'])>1e-5*max(1,abs(expected)):
        result.validation['valid']=False
        result.validation['violations'].append('Solver objective disagrees with replay accounting')
    result.feasible=result.validation['valid']
    result.optimal=result.feasible and raw.status==0
    result.objective_cost=result.accounting['total_cost']
    if not result.feasible: result.status='invalid_solution'
    return result
