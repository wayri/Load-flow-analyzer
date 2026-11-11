"""Run with python -m nrlarge --help. No worker is launched by a solve command."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from . import __version__
from .examples import unbalanced_example, synthetic, convert_balanced
from .model import Network, read_case, write_case
from .linear import LocalBackend
from .reports import empty_output, write_reports, preflight
from .solver import solve


def parser():
    p = argparse.ArgumentParser(description='NRlarge: sparse unbalanced steady-state load flow; coordinator-limited cluster prototype')
    p.add_argument('--version',action='version',version=__version__)
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('economics',help='Plan, optimize, check and replan economic dispatch; economics --help for commands')
    sub.add_parser('hardware',help='Probe actual OpenCL FP64/OpenGL compute devices')
    example = sub.add_parser('example',help='Write an editable unbalanced four-wire wye/delta case')
    example.add_argument('--out',required=True)
    convert = sub.add_parser('convert-balanced',help='Expand existing balanced CSV data into three equivalent phases')
    convert.add_argument('--buses',required=True); convert.add_argument('--branches',required=True)
    convert.add_argument('--out',required=True); convert.add_argument('--base-mva',type=float,default=100)
    worker = sub.add_parser('worker',help='Explicit numeric worker; verified TLS required off loopback')
    worker.add_argument('--bind',default='127.0.0.1'); worker.add_argument('--port',type=int,default=8843)
    worker.add_argument('--token-file',required=True); worker.add_argument('--cert'); worker.add_argument('--key')
    worker.add_argument('--memory-gib',type=float,default=1); worker.add_argument('--lease-seconds',type=float,default=300)
    worker.add_argument('--threads',type=int,default=1)
    worker.add_argument('--accelerator',choices=('cpu','opencl','opengl'),default='cpu')
    worker.add_argument('--allow-loopback-http',action='store_true',help='Local test transport only; cannot expose remote HTTP')
    for name in ('solve','benchmark'):
        q = sub.add_parser(name,help='Solve supplied data' if name == 'solve' else 'In-memory SYNTHETIC customer study, no full export')
        if name == 'solve':
            q.add_argument('--case',required=True); q.add_argument('--out',required=True)
            q.add_argument('--feeder',help='Restrict sampled plots to one feeder; CSV still includes all data')
            q.add_argument('--plot-points',type=int,default=512); q.add_argument('--max-output-mib',type=float,default=512)
        else:
            q.add_argument('--customers',type=int,default=1000)
        q.add_argument('--threads',type=int,default=max(1,min(4,os.cpu_count() or 1)))
        q.add_argument('--linear',choices=('gmres','direct'),default='gmres')
        q.add_argument('--accelerator',choices=('cpu','opencl','opengl'),default='cpu')
        q.add_argument('--memory-gib',type=float,default=4); q.add_argument('--restart',type=int,default=30)
        q.add_argument('--ordering',choices=('input','rcm'),default='rcm',help='Graph ordering before contiguous paired-row partitioning')
        q.add_argument('--tolerance',type=float,default=1e-8); q.add_argument('--max-iterations',type=int,default=100)
        q.add_argument('--workers',help='JSON file containing explicit HTTPS worker origins (maximum20)')
        q.add_argument('--token-file'); q.add_argument('--ca-file'); q.add_argument('--worker-timeout',type=float,default=60)
        q.add_argument('--allow-loopback-http',action='store_true')
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'economics':
        from .economics.__main__ import main as economics_main
        return economics_main(argv[1:])
    started = time.perf_counter()
    args = parser().parse_args(argv)
    try:
        if args.command == 'hardware':
            from .accelerators import accelerator_info
            print(json.dumps(accelerator_info(),indent=2)); return 0
        if args.command in ('example','convert-balanced'):
            out = empty_output(args.out)
            case = unbalanced_example() if args.command == 'example' else convert_balanced(args.buses,args.branches,args.base_mva)
            Network(case)
            write_case(case,out); print(f'Editable inputs: {out.absolute()}'); return 0
        if args.command == 'worker':
            from .cluster import WorkerState, make_server, token_from_file
            state = WorkerState(args.accelerator,args.memory_gib,args.lease_seconds,args.threads)
            server = make_server(args.bind,args.port,token_from_file(args.token_file),state,args.cert,args.key,args.allow_loopback_http)
            print(f'Numeric worker listening on {args.bind}:{server.server_address[1]}; {args.accelerator}; threads={args.threads}',flush=True)
            try:
                while True:
                    server.handle_request()
                    if state.job is not None and time.monotonic()-state.touched > state.lease:
                        state.close()
            except KeyboardInterrupt:
                return 0
            finally:
                state.close(); server.server_close()
        if args.command == 'solve':
            empty_output(args.out)
            case = read_case(args.case,memory_gib=args.memory_gib,restart=args.restart)
        else:
            case = synthetic(args.customers,args.memory_gib,args.restart)
        net = Network(case,args.memory_gib,args.restart,args.ordering)
        if args.command == 'solve':
            preflight(net,args.out,args.plot_points,args.max_output_mib)
            if args.feeder is not None and not any(node.feeder == args.feeder for node in case.nodes):
                raise ValueError('Selected feeder has no conductors')
        setup_seconds = time.perf_counter()-started
        execution = {'mode':'local','threads':args.threads,'linear':args.linear,'accelerator':args.accelerator}
        execution['ordering'] = args.ordering
        if args.workers:
            if args.linear != 'gmres' or not args.token_file or args.accelerator != 'cpu':
                raise ValueError('Cluster requires --linear gmres and --token-file; select accelerators on each worker')
            from .cluster import ClusterBackend, token_from_file
            endpoints = json.loads(Path(args.workers).read_text(encoding='utf-8'))
            if not isinstance(endpoints,list) or any(not isinstance(v,str) for v in endpoints):
                raise ValueError('Worker config must be a JSON list of HTTPS origins')
            backend = ClusterBackend(endpoints,token_from_file(args.token_file),args.ca_file,args.restart,args.worker_timeout,args.allow_loopback_http)
            execution = {'mode':'cluster','workers':len(endpoints),'accelerators':'configured per worker','linear':'gmres','ordering':args.ordering}
        else:
            backend = LocalBackend(args.threads,args.accelerator,args.restart,args.linear)
        try:
            result = solve(net,backend,args.tolerance,args.max_iterations)
        finally:
            backend.close()
        if args.command == 'solve':
            summary = write_reports(net,result,args.out,execution,args.feeder,args.plot_points,args.max_output_mib)
        else:
            summary = {'dataset':'SYNTHETIC, not national data','customers':args.customers,'nodes':len(case.nodes),
                       'y_nnz':net.y.nnz,'estimated_peak_gib':net.estimated_bytes/2**30,'seconds':result.seconds,
                       'iterations':len(result.history)-1,'converged':result.converged,'reason':result.reason,
                       'residual':result.history[-1]['residual'],'power_error':result.history[-1]['power_error'],
                       'execution':execution}
        summary['setup_seconds'] = setup_seconds
        summary['total_seconds'] = time.perf_counter()-started
        print(json.dumps(summary,indent=2,allow_nan=False))
        return 0 if result.converged else 2
    except (Exception,MemoryError) as error:
        print(f'NRlarge error: {error}',file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
