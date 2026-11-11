"""Reproduce the small synthetic economics gallery in a new output directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from .examples import example_case
from .grid import check_grid
from .io import save_case, save_dispatch
from .management import baseline_case, state_at, replan
from .optimizer import optimize
from .planning import plan_sizes
from .reports import write_reports


def _json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build(directory):
    out=Path(directory)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Gallery requires a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]; cases={}; dispatches={}
    for role in ('battery','generator','discom','aggregator'):
        case=example_case(role); result=optimize(case); baseline=optimize(baseline_case(case))
        folder=out/role
        save_case(case,folder/'inputs'); save_dispatch(result,folder/'dispatch.json')
        save_dispatch(baseline,folder/'baseline-dispatch.json')
        summary=write_reports(case,result,folder,baseline=baseline)
        if not result.optimal or not result.validation['valid']:
            raise ValueError(f'{role} example failed acceptance')
        summaries.append({'role':role,'net_value':result.accounting['net_profit'],
                          'battery_benefit':summary['baseline'].get('baseline_cost_minus_dispatch_cost')})
        cases[role]=case; dispatches[role]=result
    # A projected six-hour operating boundary exercises the rolling plan.
    rolling=out/'replan'; rolling.mkdir()
    state=state_at(cases['battery'],dispatches['battery'],6)
    _json(rolling/'projected-state.json',state)
    remaining=replan(cases['battery'],state,state['boundary'])
    result=optimize(remaining); baseline=optimize(baseline_case(remaining))
    save_case(remaining,rolling/'inputs'); save_dispatch(result,rolling/'dispatch.json')
    save_dispatch(baseline,rolling/'baseline-dispatch.json')
    write_reports(remaining,result,rolling,baseline=baseline)
    # Capacity comparisons retain compact receipts; full candidate charts are
    # available through the public `plan` CLI.
    sizing=out/'sizing'; sizing.mkdir()
    fields=['candidate_id','battery_id','capacity_mwh','charge_max_mw','discharge_max_mw','horizon_fixed_cost']
    with (sizing/'candidates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(fields)
        writer.writerows([['none','B1',0,0,0,0],['small','B1',3,1,1,5],
                          ['reference','B1',6,2,2,10],['large','B1',9,3,3,20]])
    comparison,schedules=plan_sizes(cases['battery'],sizing/'candidates.csv')
    _json(sizing/'planning.json',comparison)
    with (sizing/'planning.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison['candidates'][0])); writer.writeheader()
        writer.writerows(comparison['candidates'])
    for k,(case,result) in enumerate(schedules):
        save_case(case,sizing/f'candidate-{k:03d}'/'inputs')
        save_dispatch(result,sizing/f'candidate-{k:03d}'/'dispatch.json')
    # A passive two-conductor feeder supplies the station. Small finite R
    # produces an explicit difference between forecast and actual interchange.
    from ..model import Case as GridCase, Node, add_series, write_case
    ac=out/'ac-replay'; ac.mkdir()
    nodes=[Node('source.A',type='SLACK'),Node('source.N',phase='N',type='SLACK',vm=0),
           Node('station.A'),Node('station.N',phase='N',vm=0)]
    stamps=[]
    add_series(stamps,'feeder',['source.A','source.N'],['station.A','station.N'],np.diag([.0005+.001j,.0005+.001j]))
    write_case(GridCase(nodes,stamps,base_mva=3),ac/'passive-feeder')
    port={'from':'station.A','to':'station.N','weight':1,'q_per_p':0}
    config={'sites':{'battery-meter':{'case':'passive-feeder','ports':{'fixed_load':[port],'renewable':[port],'battery:B1':[port]},
        'voltage_min_pu':.95,'voltage_max_pu':1.05,'interchange_tolerance_mw':.01,
        'terminal_current_limits_pu':{'feeder':2.1}}}}
    _json(ac/'mapping.json',config)
    ac_result=check_grid(cases['battery'],dispatches['battery'],config,ac)
    _json(ac/'grid-check.json',ac_result)
    strict=json.loads(json.dumps(config)); strict['sites']['battery-meter']['interchange_tolerance_mw']=1e-5
    _json(ac/'mapping-strict.json',strict)
    _json(ac/'grid-check-strict.json',check_grid(cases['battery'],dispatches['battery'],strict,ac))
    if not result.feasible or not ac_result['passed']:
        raise ValueError('Rolling/sizing or configured AC example failed')
    text='# Synthetic economic dispatch examples\n\nAll prices and forecasts are synthetic CUR inputs over one 24-hour horizon.\n\n'
    text+='| Role | Net modeled value (CUR) | Benefit from storage (CUR) |\n|---|---:|---:|\n'
    for r in summaries:
        text+=f"| [{r['role']}]({r['role']}/README.md) | {r['net_value']:.4f} | {r['battery_benefit']:.4f} |\n"
    text+='\nEach role includes reloadable inputs, dispatch/cash-flow CSVs, JSON receipts, ASCII asset diagrams and PNG/SVG plots. All four schedules passed independent replay and solved to the configured MILP gap.\n\n'
    text+='[Projected six-hour replan](replan/README.md) carries SOC and the remaining discharge budget. [Battery sizing](sizing/planning.json) compares four user-cost candidates; their numbered folders preserve the input/schedule receipts.\n\n'
    text+='[AC replay](ac-replay/grid-check.json) checks a synthetic passive two-wire feeder at a 0.01 MW interchange tolerance. The [strict-tolerance replay](ac-replay/grid-check-strict.json) identifies the unforecast resistive losses with the same converged schedule.\n\n'
    text+='![Battery dispatch](battery/dispatch.png)\n\n![Battery stored energy](battery/soc.png)\n\n![Battery modeled cash](battery/cash.png)\n\n'
    text+='Reproduce in a new directory: `python -m nrlarge.economics.study --out my-economic-studies`. See the [user guide](../../docs/economics.md) for checking receipts and operating with measured state. `manifest.json` records producer versions and SHA-256 hashes.\n'
    (out/'README.md').write_text(text,encoding='utf-8')
    source=Path(__file__).resolve().parent
    manifest={'schema':1,'inputs':'Synthetic examples; no live tariffs or telemetry',
              'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
              'producer_sha256_lf':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(source.glob('*.py'))},
              'artifacts':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file()}}
    _json(out/'manifest.json',manifest)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    args=p.parse_args(); print(json.dumps(build(args.out),indent=2))


if __name__=='__main__':
    main()
