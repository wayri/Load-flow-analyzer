"""Independent physical and monetary replay of public dispatch arrays."""
import math
import numpy as np


def validate_dispatch(case, dispatch, tolerance=1e-6):
    case.validate()
    if not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Positive finite replay tolerance required')
    violations=[]; n=len(next(iter(case.periods.values())))
    dt=np.array([r.duration_h for r in next(iter(case.periods.values()))]); starts=np.r_[0,np.cumsum(dt)[:-1]]
    def fail(label):
        if len(violations)<100: violations.append(label)
    def check(condition,label):
        if not np.all(condition): fail(label)
    def close(a,b,label):
        check(np.abs(np.asarray(a)-np.asarray(b))<=tolerance*np.maximum(1,np.maximum(np.abs(a),np.abs(b))),label)
    def array(group,key,name,length=n):
        try:
            value=np.asarray(group[key][name],dtype=float)
            if value.shape!=(length,) or not np.all(np.isfinite(value)): raise ValueError()
            return value
        except (KeyError,ValueError,TypeError):
            fail(f'{key}.{name}: expected {length} finite values')
            return np.zeros(length)
    def bounds(a,lo,hi,label):
        check(a>=lo-tolerance,label+' below limit'); check(a<=hi+tolerance,label+' above limit')
    def binary(a,label):
        bounds(a,0,1,label); close(a,np.round(a),label+' must be binary')
    def keys(group,items,label):
        if not isinstance(group,dict) or set(group)!={x.id for x in items}:
            fail(label+' IDs differ from case')
    for name,items in (('site',case.sites),('batteries',case.batteries),('generators',case.generators),('flexible',case.flexible_loads)):
        keys(getattr(dispatch,name),items,name)
    bats={}; gens={}; flex={}; sites={}
    for b in case.batteries:
        ch=array(dispatch.batteries,b.id,'charge_mw'); dis=array(dispatch.batteries,b.id,'discharge_mw')
        energy=array(dispatch.batteries,b.id,'energy_mwh',n+1); mode=array(dispatch.batteries,b.id,'charge_mode')
        binary(mode,b.id+' charge_mode')
        bounds(ch,0,b.charge_max_mw,b.id+' charge'); bounds(dis,0,b.discharge_max_mw,b.id+' discharge')
        check(ch<=b.charge_max_mw*mode+tolerance,b.id+' charge exclusivity')
        check(dis<=b.discharge_max_mw*(1-mode)+tolerance,b.id+' discharge exclusivity')
        hi=b.capacity_mwh if b.soc_max_mwh is None else b.soc_max_mwh
        bounds(energy,b.soc_min_mwh,hi,b.id+' SOC')
        close(energy[0],b.initial_soc_mwh,b.id+' initial SOC')
        close(energy[-1],b.initial_soc_mwh if b.terminal_soc_mwh is None else b.terminal_soc_mwh,b.id+' terminal SOC')
        close(energy[1:],energy[:-1]*(1-b.self_discharge_per_h)**dt+b.eta_charge*ch*dt-dis*dt/b.eta_discharge,b.id+' energy conservation')
        if b.max_discharge_mwh is not None:
            check(np.dot(dis,dt)<=b.max_discharge_mwh+tolerance,b.id+' discharge budget')
        bats[b.id]=(ch,dis)
    for g in case.generators:
        p=array(dispatch.generators,g.id,'p_mw'); on=array(dispatch.generators,g.id,'on')
        startup=array(dispatch.generators,g.id,'startup'); shutdown=array(dispatch.generators,g.id,'shutdown')
        for a,name in ((on,'on'),(startup,'startup'),(shutdown,'shutdown')): binary(a,g.id+' '+name)
        bounds(p,g.p_min_mw*on,g.p_max_mw*on,g.id+' output')
        prev_u=np.r_[g.initial_on,on[:-1]]; prev_p=np.r_[g.initial_p_mw,p[:-1]]
        close(on-prev_u,startup-shutdown,g.id+' commitment transition')
        check(startup+shutdown<=1+tolerance,g.id+' startup/shutdown exclusivity')
        ru=np.full(n,g.p_max_mw) if g.ramp_up_mw_per_h is None else dt*g.ramp_up_mw_per_h
        rd=np.full(n,g.p_max_mw) if g.ramp_down_mw_per_h is None else dt*g.ramp_down_mw_per_h
        su=g.p_max_mw if g.startup_ramp_mw is None else g.startup_ramp_mw
        sh=g.p_max_mw if g.shutdown_ramp_mw is None else g.shutdown_ramp_mw
        check(p-prev_p<=ru*prev_u+su*startup+tolerance,g.id+' ramp up')
        check(prev_p-p<=rd*on+sh*shutdown+tolerance,g.id+' ramp down')
        for t in range(n):
            if startup[t]>.5:
                check(on[(starts>=starts[t]) & (starts<starts[t]+g.min_up_h-1e-9)]>=1-tolerance,g.id+' minimum up time')
            if shutdown[t]>.5:
                check(on[(starts>=starts[t]) & (starts<starts[t]+g.min_down_h-1e-9)]<=tolerance,g.id+' minimum down time')
        remaining=(g.min_up_h if g.initial_on else g.min_down_h)-g.initial_state_h
        close(on[starts<remaining-1e-9],g.initial_on,g.id+' initial minimum state duration')
        gens[g.id]=(p,on,startup)
    for f in case.flexible_loads:
        p=array(dispatch.flexible,f.id,'p_mw'); end=n if f.end_slot is None else f.end_slot
        bounds(p,0,f.p_max_mw,f.id+' flexible power')
        outside=np.ones(n,dtype=bool); outside[f.start_slot:end]=False
        close(p[outside],0,f.id+' outside service window')
        close(np.dot(p,dt),f.energy_mwh,f.id+' flexible energy obligation')
        flex[f.id]=p
    lines=[]; peak_cost=0; peak_by_site={}
    for s in case.sites:
        imp=array(dispatch.site,s.id,'import_mw'); exp=array(dispatch.site,s.id,'export_mw')
        ren=array(dispatch.site,s.id,'renewable_used_mw'); mode=array(dispatch.site,s.id,'grid_mode')
        binary(mode,s.id+' grid_mode')
        bounds(imp,0,s.import_limit_mw,s.id+' import'); bounds(exp,0,s.export_limit_mw,s.id+' export')
        check(imp<=mode*s.import_limit_mw+tolerance,s.id+' import exclusivity')
        check(exp<=(1-mode)*s.export_limit_mw+tolerance,s.id+' export exclusivity')
        bounds(ren,0,np.array([r.renewable_mw for r in case.periods[s.id]]),s.id+' renewable')
        peak=max(s.historical_peak_mw,float(np.max(imp))); peak_by_site[s.id]=peak
        peak_cost+=(peak-s.historical_peak_mw)*s.demand_charge_per_mw
        for t,r in enumerate(case.periods[s.id]):
            charge=sum(bats[b.id][0][t] for b in case.batteries if b.site==s.id)
            discharge=sum(bats[b.id][1][t] for b in case.batteries if b.site==s.id)
            generation=sum(gens[g.id][0][t] for g in case.generators if g.site==s.id)
            flexible_load=sum(flex[f.id][t] for f in case.flexible_loads if f.site==s.id)
            close(imp[t]+ren[t]+generation+discharge,exp[t]+r.load_mw+r.loss_mw+charge+flexible_load,f'{s.id} slot {t} power balance')
            row={'site':s.id,'slot':t,'time':r.time,'duration_h':r.duration_h,
                 'import_cost':imp[t]*r.buy_price*r.duration_h,
                 'export_revenue':exp[t]*r.sell_price*r.duration_h,
                 'retail_revenue':r.load_mw*r.retail_price*r.duration_h,
                 'generator_cost':sum((gens[g.id][0][t]*g.marginal_cost_per_mwh+gens[g.id][1][t]*g.no_load_cost_per_h)*r.duration_h for g in case.generators if g.site==s.id),
                 'startup_cost':sum(gens[g.id][2][t]*g.startup_cost for g in case.generators if g.site==s.id),
                 'degradation_cost':sum(bats[b.id][1][t]*r.duration_h*b.degradation_cost_per_mwh for b in case.batteries if b.site==s.id),
                 'curtailment_cost':(r.renewable_mw-ren[t])*r.duration_h*case.renewable_curtail_cost_per_mwh,
                 'flexible_revenue':sum(flex[f.id][t]*r.duration_h*f.value_per_mwh for f in case.flexible_loads if f.site==s.id)}
            row['total_cost']=sum(row[k] for k in ('import_cost','generator_cost','startup_cost','degradation_cost','curtailment_cost'))-sum(row[k] for k in ('export_revenue','retail_revenue','flexible_revenue'))
            row['net_profit']=-row['total_cost']; lines.append(row)
    totals={key:float(sum(row[key] for row in lines)) for key in ('import_cost','export_revenue','retail_revenue','generator_cost','startup_cost','degradation_cost','curtailment_cost','flexible_revenue')}
    total=float(sum(row['total_cost'] for row in lines)+peak_cost+case.fixed_horizon_cost)
    accounting={**totals,'demand_charge':float(peak_cost),'fixed_horizon_cost':case.fixed_horizon_cost,
                'peak_import_mw':peak_by_site,'total_cost':total,'net_profit':-total,'currency':case.currency,'lines':lines}
    def receipt_matches(declared,expected):
        if isinstance(expected,dict):
            return (isinstance(declared,dict) and set(declared)==set(expected) and
                    all(receipt_matches(declared[k],v) for k,v in expected.items()))
        if isinstance(expected,list):
            return isinstance(declared,list) and len(declared)==len(expected) and all(receipt_matches(a,b) for a,b in zip(declared,expected))
        if isinstance(expected,(int,float)):
            return (isinstance(declared,(int,float)) and not isinstance(declared,bool) and math.isfinite(declared)
                    and abs(declared-expected)<=tolerance*max(1,abs(expected),abs(declared)))
        return declared==expected
    if dispatch.accounting and not receipt_matches(dispatch.accounting,accounting):
        fail('Declared accounting receipt disagrees with independent replay')
    if dispatch.objective_cost is not None and not receipt_matches(dispatch.objective_cost,total):
        fail('Declared objective cost disagrees with independent replay')
    return {'valid':not violations,'violations':violations,'tolerance':tolerance,'accounting':accounting}
