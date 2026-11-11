"""Replay a dispatch on passive unbalanced AC networks at each site's PCC.

The economic MILP uses supplied loss forecasts; this postcheck reports actual
slack interchange, voltage and configured element-terminal current limits.
"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from ..model import Network, Load, read_case, evaluate
from ..solver import solve


def check_grid(case, dispatch, mapping, base_directory='.'):
    from .validation import validate_dispatch
    replay=validate_dispatch(case,dispatch)
    if not replay['valid']:
        raise ValueError('Grid replay requires an independently valid dispatch')
    if set(mapping.get('sites',{}))!={s.id for s in case.sites}:
        raise ValueError('Grid mapping requires every dispatch site')
    results=[]; overall=True
    for site in case.sites:
        cfg=mapping['sites'][site.id]
        template=read_case(Path(base_directory)/cfg['case'])
        if template.loads or any(n.type not in ('SLACK','PQ') or n.p or n.q for n in template.nodes):
            raise ValueError('AC templates require passive PQ conductors and SLACK sources, with zero existing injections/loads')
        nodes={x.id:x for x in template.nodes}
        resources={'fixed_load':None,'renewable':None}
        resources.update({'battery:'+b.id:b for b in case.batteries if b.site==site.id})
        resources.update({'generator:'+g.id:g for g in case.generators if g.site==site.id})
        resources.update({'flexible:'+f.id:f for f in case.flexible_loads if f.site==site.id})
        if set(cfg['ports'])!=set(resources):
            raise ValueError('Map fixed_load, renewable and every site asset exactly once')
        for key,ports in cfg['ports'].items():
            if not ports or not math.isclose(sum(p['weight'] for p in ports),1,abs_tol=1e-9):
                raise ValueError(f'{key} port weights must sum to one')
            for p in ports:
                if p['from'] not in nodes or p['to'] not in nodes or p['from']==p['to']:
                    raise ValueError('Ports require distinct explicit phase/neutral conductor IDs')
                if not all(math.isfinite(p.get(k,0)) for k in ('weight','q_per_p')) or p['weight']<0:
                    raise ValueError('Port weights and reactive factors must be finite; weights nonnegative')
        vmin=float(cfg.get('voltage_min_pu',.9)); vmax=float(cfg.get('voltage_max_pu',1.1))
        tol=float(cfg.get('interchange_tolerance_mw',1e-4))
        if not all(math.isfinite(x) for x in (vmin,vmax,tol)) or not 0<vmin<=vmax or tol<0:
            raise ValueError('Invalid AC operating limits')
        ratings=cfg.get('terminal_current_limits_pu',{})
        elements={stamp[0] for stamp in template.stamps}
        if set(ratings)-elements or any(not math.isfinite(v) or v<=0 for v in ratings.values()):
            raise ValueError('Current limits require known element IDs and positive finite pu ratings')
        for t,row in enumerate(case.periods[site.id]):
            network_case=deepcopy(template)
            powers={'fixed_load':row.load_mw,'renewable':-dispatch.site[site.id]['renewable_used_mw'][t]}
            powers.update({'battery:'+b.id:dispatch.batteries[b.id]['charge_mw'][t]-dispatch.batteries[b.id]['discharge_mw'][t] for b in case.batteries if b.site==site.id})
            powers.update({'generator:'+g.id:-dispatch.generators[g.id]['p_mw'][t] for g in case.generators if g.site==site.id})
            powers.update({'flexible:'+f.id:dispatch.flexible[f.id]['p_mw'][t] for f in case.flexible_loads if f.site==site.id})
            for resource,ports in cfg['ports'].items():
                for k,p in enumerate(ports):
                    power=powers[resource]*p['weight']/(template.base_mva/3)
                    network_case.loads.append(Load(f'dispatch:{resource}:{k}',p['from'],p['to'],power,power*p.get('q_per_p',0)))
            net=Network(network_case); solved=solve(net)
            entry={'site':site.id,'slot':t,'time':row.time,'converged':solved.converged,'reason':solved.reason,'passed':False}
            if solved.converged:
                _,actual,_=evaluate(net,solved.voltage,solved.types,jacobian=False)
                slack=float(sum(actual[k].real for k,node in enumerate(template.nodes) if node.type=='SLACK')*(template.base_mva/3))
                scheduled=dispatch.site[site.id]['import_mw'][t]-dispatch.site[site.id]['export_mw'][t]
                port_voltages=[abs(solved.voltage[net.index[p['from']]]-solved.voltage[net.index[p['to']]]) for ports in cfg['ports'].values() for p in ports]
                currents={}
                for element in ratings:
                    terminals={}
                    for name,a,b,y in template.stamps:
                        if name==element:
                            terminals[a]=terminals.get(a,0j)+y*solved.voltage[net.index[b]]
                    currents[element]=max((abs(v) for v in terminals.values()),default=0)
                passed=(min(port_voltages)>=vmin and max(port_voltages)<=vmax and
                        abs(slack-scheduled)<=tol and -site.export_limit_mw-tol<=slack<=site.import_limit_mw+tol and
                        all(currents[k]<=ratings[k]+1e-9 for k in ratings))
                entry.update(passed=bool(passed),actual_pcc_mw=slack,scheduled_pcc_mw=scheduled,
                    interchange_error_mw=slack-scheduled,voltage_min_pu=float(min(port_voltages)),voltage_max_pu=float(max(port_voltages)),
                    terminal_current_max_pu=currents,thermal_limits_checked=bool(ratings))
            overall=overall and entry['passed']; results.append(entry)
    return {'passed':bool(overall),'scope':'Independent AC replay using supplied port mapping and limits',
            'economic_dispatch_ac_constrained':False,'intervals':results}
