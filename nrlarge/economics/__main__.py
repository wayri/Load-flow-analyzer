"""Runnable economic planning, dispatch, independent checking and replanning."""
import argparse
import csv
import json
from pathlib import Path
import sys
from .io import load_case, save_case, load_dispatch, save_dispatch, _read_json
from .optimizer import optimize
from .validation import validate_dispatch


def _json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8') as stream:
        stream.write(payload)


def _out(value):
    path=Path(value)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path


def parser():
    p=argparse.ArgumentParser(description='Market-neutral multi-site economic dispatch and planning (MW/MWh)')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('example',help='Create editable synthetic forecast and asset CSV inputs')
    q.add_argument('--role',choices=('battery','generator','discom','aggregator'),default='battery')
    q.add_argument('--out',required=True)
    for command in ('optimize','replan','plan'):
        q=sub.add_parser(command)
        q.add_argument('--case',required=True); q.add_argument('--out',required=True)
        q.add_argument('--time-limit',type=float,default=60)
        if command=='plan':
            q.add_argument('--candidates',required=True)
        else:
            q.add_argument('--baseline',action='store_true',help='Reoptimize the same case with batteries removed')
            q.add_argument('--grid-mapping',help='Optional unbalanced AC postcheck configuration JSON')
        if command=='replan':
            q.add_argument('--state',required=True); q.add_argument('--from-time')
    q=sub.add_parser('check',help='Independent replay of saved dispatch; returns exit2 if invalid')
    q.add_argument('--case',required=True); q.add_argument('--result',required=True); q.add_argument('--out')
    q=sub.add_parser('state',help='Export projected state after N completed slots; replace with measurements for operation')
    q.add_argument('--case',required=True); q.add_argument('--result',required=True)
    q.add_argument('--after-slot',type=int,required=True); q.add_argument('--out',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='example':
            from .examples import example_case
            save_case(example_case(args.role),args.out)
            print(f'Editable {args.role} inputs: {Path(args.out).absolute()}'); return 0
        case=load_case(args.case)
        if args.command=='check':
            result=load_dispatch(args.result)
            report=validate_dispatch(case,result)
            if args.out: _json(args.out,report)
            print(json.dumps({'valid':report['valid'],'violations':report['violations'],'total_cost':report['accounting']['total_cost'],'currency':case.currency},indent=2))
            return 0 if report['valid'] else 2
        if args.command=='state':
            from .management import state_at
            result=load_dispatch(args.result)
            report=validate_dispatch(case,result)
            if not report['valid']: raise ValueError('State export requires independently valid dispatch')
            _json(args.out,state_at(case,result,args.after_slot))
            print(f'Projected state: {Path(args.out).absolute()}'); return 0
        from .reports import write_reports
        if args.command=='plan':
            from .planning import plan_sizes
            comparison,schedules=plan_sizes(case,args.candidates,args.time_limit)
            out=_out(args.out)
            _json(out/'planning.json',comparison)
            with (out/'planning.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(comparison['candidates'][0]))
                writer.writeheader(); writer.writerows(comparison['candidates'])
            for k,(candidate,result) in enumerate(schedules):
                directory=out/f'candidate-{k:03d}'; directory.mkdir()
                save_case(candidate,directory/'inputs'); save_dispatch(result,directory/'dispatch.json')
                write_reports(candidate,result,directory)
            print(json.dumps(comparison,indent=2)); return 0 if comparison['best_candidate'] is not None else 2
        state=None
        if args.command=='replan':
            from .management import replan
            state=_read_json(args.state)
            case=replan(case,state,args.from_time or state['boundary'])
        result=optimize(case,time_limit=args.time_limit)
        baseline=None
        if args.baseline:
            from .management import baseline_case
            baseline=optimize(baseline_case(case),time_limit=args.time_limit)
        grid=None
        if args.grid_mapping and result.feasible:
            from .grid import check_grid
            mapping_path=Path(args.grid_mapping)
            mapping=_read_json(mapping_path)
            grid=check_grid(case,result,mapping,mapping_path.parent)
        out=_out(args.out)
        save_case(case,out/'inputs'); save_dispatch(result,out/'dispatch.json')
        if baseline is not None: save_dispatch(baseline,out/'baseline-dispatch.json')
        if state is not None: _json(out/'starting-state.json',state)
        summary=write_reports(case,result,out,baseline=baseline,grid_check=grid)
        print(json.dumps({'status':result.status,'optimal':result.optimal,'feasible':result.feasible,
                          'total_cost':result.accounting.get('total_cost'),'net_profit':result.accounting.get('net_profit'),
                          'currency':case.currency,'grid_passed':None if grid is None else grid['passed'],
                          'out':str(out.absolute())},indent=2))
        return 0 if result.feasible and (grid is None or grid['passed']) else 2
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'economics: {exc}',file=sys.stderr); return 1


if __name__=='__main__':
    raise SystemExit(main())
