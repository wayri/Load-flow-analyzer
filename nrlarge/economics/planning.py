"""Compare battery sizes over the supplied forecast horizon and user costs."""
from copy import deepcopy
import csv
import math
from .optimizer import optimize


def plan_sizes(case, candidates_file, time_limit=60):
    case.validate()
    with open(candidates_file,newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        expected=['candidate_id','battery_id','capacity_mwh','charge_max_mw','discharge_max_mw','horizon_fixed_cost']
        if reader.fieldnames!=expected:
            raise ValueError('Candidate CSV header must be '+','.join(expected))
        rows=[]
        for row in reader:
            if len(rows)>=50: raise ValueError('At most 50 sizing candidates per run')
            if None in row or any(v is None for v in row.values()): raise ValueError('Malformed candidate row')
            rows.append(row)
    if not rows or len({r['candidate_id'] for r in rows})!=len(rows) or any(not r['candidate_id'].strip() for r in rows):
        raise ValueError('Sizing requires unique nonempty candidate IDs')
    candidates=[]
    for row in rows:
        modeled=deepcopy(case)
        matches=[b for b in modeled.batteries if b.id==row['battery_id']]
        if len(matches)!=1: raise ValueError('Each sizing row must select an existing battery ID')
        b=matches[0]
        end=b.initial_soc_mwh if b.terminal_soc_mwh is None else b.terminal_soc_mwh
        if abs(end-b.initial_soc_mwh)>1e-9 or b.capacity_mwh<=0:
            raise ValueError('Sizing requires a positive reference capacity and cyclic initial/terminal SOC')
        capacity,charge,discharge,cost=[float(row[k]) for k in expected[2:]]
        if not all(math.isfinite(v) and v>=0 for v in (capacity,charge,discharge,cost)):
            raise ValueError('Sizing capacities, powers and horizon costs must be finite and nonnegative')
        if capacity==0:
            if charge or discharge: raise ValueError('Zero-capacity candidates require zero power')
            modeled.batteries.remove(b)
        else:
            ratio=capacity/b.capacity_mwh
            for name in ('soc_min_mwh','soc_max_mwh','initial_soc_mwh','terminal_soc_mwh','max_discharge_mwh'):
                value=getattr(b,name)
                if value is not None: setattr(b,name,value*ratio)
            b.capacity_mwh=capacity; b.charge_max_mw=charge; b.discharge_max_mw=discharge
        modeled.fixed_horizon_cost+=cost
        candidates.append((row,modeled))
    outcomes=[]; schedules=[]
    for row,modeled in candidates:
        result=optimize(modeled,time_limit=time_limit)
        total=result.accounting.get('total_cost') if result.feasible else None
        outcomes.append({**row,'status':result.status,'optimal':result.optimal,'feasible':result.feasible,
                         'total_horizon_cost':total,'net_horizon_value':-total if total is not None else None})
        schedules.append((modeled,result))
    feasible=[r for r in outcomes if r['feasible']]
    ranked=sorted(feasible,key=lambda r:r['total_horizon_cost'])
    return {'scope':'Supplied forecast horizon; candidate cost is charged once over this horizon',
            'currency':case.currency,'best_candidate':ranked[0]['candidate_id'] if ranked else None,
            'all_feasible_candidates_optimal':all(r['optimal'] for r in feasible),
            'ranking':[r['candidate_id'] for r in ranked],'candidates':outcomes},schedules
