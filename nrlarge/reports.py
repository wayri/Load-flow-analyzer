"""Stream full CSV/ASCII outputs; bound chart work independently of network size."""
import csv
from html import escape
from itertools import groupby
import json
from pathlib import Path
import shutil
import numpy as np
from .model import evaluate, write_case


def empty_output(directory):
    path = Path(directory)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output directory must be new or empty: {path}')
    return path


def shards(directory, name, fields, rows, limit=100000):
    count = 0; stream = None
    try:
        for row in rows:
            if count % limit == 0:
                if stream is not None: stream.close()
                stream = (directory/f'{name}-{count//limit:05d}.csv').open('w',encoding='utf-8',newline='')
                writer = csv.writer(stream); writer.writerow(fields)
            writer.writerow(row); count += 1
        if stream is None:
            with (directory/f'{name}-00000.csv').open('w',encoding='utf-8',newline='') as f:
                csv.writer(f).writerow(fields)
    finally:
        if stream is not None: stream.close()
    return count


def svg(path, title, values, labels=None, limit=512):
    n = len(values)
    sample = np.unique(np.linspace(0,max(0,n-1),min(n,limit),dtype=int))
    selected = np.asarray(values)[sample] if n else np.array([0.0])
    low,high = min(0,float(selected.min())),max(0,float(selected.max()))
    span = max(high-low,1e-12); low -= span*.08; high += span*.08
    def x(k): return 75 + k*850/max(1,len(sample)-1)
    def y(v): return 340-(float(v)-low)/(high-low)*260
    with Path(path).open('w',encoding='utf-8') as out:
        out.write('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430" viewBox="0 0 1000 430"><rect width="1000" height="430" fill="white"/><g font-family="sans-serif" fill="#18283b">')
        out.write(f'<text x="30" y="30" font-size="20">{escape(title)}</text>')
        out.write(f'<text x="30" y="53" font-size="12">{len(sample)} sampled of {n}; full values in CSV. x = conductor/iteration index.</text>')
        for k in range(6):
            val = low+(high-low)*k/5
            out.write(f'<path d="M70 {y(val)} H940" stroke="#ddd"/><text x="5" y="{y(val)}" font-size="11">{val:.4g}</text>')
        out.write('<polyline fill="none" stroke="#1769aa" stroke-width="1.5" points="')
        out.write(' '.join(f'{x(k)},{y(v)}' for k,v in enumerate(selected))); out.write('"/>')
        stride = max(1,len(sample)//10)
        for k in range(0,len(sample),stride):
            label = str(sample[k]) if labels is None else str(labels[sample[k]])[:40]
            out.write(f'<text transform="translate({x(k)},360) rotate(25)" font-size="10">{escape(label)}</text>')
        out.write('</g></svg>\n')


def terminal_rows(net, voltage, converged):
    for element,terms in groupby(net.case.stamps,key=lambda v:v[0]):
        currents = {}
        for _,a,b,y in terms:
            currents[a] = currents.get(a,0j)+y*voltage[net.index[b]]
        for terminal,current in currents.items():
            power = voltage[net.index[terminal]]*np.conj(current)
            yield [element,terminal,current.real,current.imag,power.real,power.imag,int(converged)]


def preflight(net, directory, max_plot_points=512, max_output_mib=512):
    directory = empty_output(directory)
    if not 2 <= max_plot_points <= 4096 or not np.isfinite(max_output_mib) or max_output_mib <= 0:
        raise ValueError('Invalid plot/output bounds')
    estimate = len(net.case.nodes)*1600+len(net.case.stamps)*500+len(net.case.loads)*1600+1024*1024
    if estimate > max_output_mib*2**20:
        raise ValueError('Estimated full report exceeds --max-output-mib; raise explicitly or use benchmark without export')
    parent = directory.absolute()
    while not parent.exists(): parent = parent.parent
    if estimate > 16*2**20 and shutil.disk_usage(parent).free-estimate < 100*2**30:
        raise OSError('Large export refused: destination must retain 100 GiB free; no storage was relocated')
    return directory


def write_reports(net, result, directory, execution, focus_feeder=None, max_plot_points=512, max_output_mib=512):
    directory = preflight(net,directory,max_plot_points,max_output_mib)
    directory.mkdir(parents=True,exist_ok=True)
    # Only create a success summary after all files are written. Failures leave
    # an explicitly incomplete directory rather than a fabricated solved receipt.
    (directory/'INCOMPLETE').write_text('Report export has not completed.\n',encoding='utf-8')
    write_case(net.case,directory/'inputs')
    _,actual,load_i = evaluate(net,result.voltage,result.types,jacobian=False)
    scale = net.case.base_mva/3
    def node_rows():
        for k,node in enumerate(net.case.nodes):
            yield [node.id,node.bus,node.phase,node.type,result.types[k],abs(result.voltage[k]),
                   np.angle(result.voltage[k],deg=True),actual[k].real,actual[k].imag,
                   actual[k].real*scale,actual[k].imag*scale,node.region,node.feeder,node.customer,int(result.converged)]
    shards(directory,'nodes','id,bus,phase,requested_type,solved_type,vm_pu,va_deg,p_generation_pu,q_generation_pu,p_generation_mw,q_generation_mvar,region,feeder,customer,converged'.split(','),node_rows())
    load_voltage = net.a @ result.voltage
    load_power = load_voltage*np.conj(load_i)
    def load_rows():
        for k,load in enumerate(net.case.loads):
            yield [load.id,load.source,load.target,load.customer,load.feeder,load.region,
                   abs(load_voltage[k]),
                   load_power[k].real,load_power[k].imag,load_power[k].real*scale,load_power[k].imag*scale,int(result.converged)]
    shards(directory,'loads','id,from,to,customer,feeder,region,terminal_vm_pu,p_pu,q_pu,p_mw,q_mvar,converged'.split(','),load_rows())
    shards(directory,'terminals','element,terminal,i_re_pu,i_im_pu,p_pu,q_pu,converged'.split(','),terminal_rows(net,result.voltage,result.converged))
    fields = ['iteration','residual','power_error','krylov_iterations','linear_residual','linear_seconds','damping','workers']
    shards(directory,'convergence',fields,([r.get(k,'') for k in fields] for r in result.history))
    totals = {}
    for k,load in enumerate(net.case.loads):
        key = (load.region,load.feeder)
        totals[key] = totals.get(key,0j)+load_power[k]*scale
    shards(directory,'feeders',['region','feeder','p_mw','q_mvar'],([*key,p.real,p.imag] for key,p in sorted(totals.items())))
    with (directory/'network.txt').open('w',encoding='utf-8') as out:
        out.write('UNBALANCED CONDUCTOR NETWORK\n'+('CONVERGED\n' if result.converged else 'NOT CONVERGED - LAST ITERATE ONLY\n'))
        out.write('Each element lists its conductor ports; mutual couplings are in inputs/stamps.csv.\n')
        for element,terms in groupby(net.case.stamps,key=lambda v:v[0]):
            ports = dict.fromkeys(a for _,a,_,_ in terms)
            out.write(f'{{{element}}}--'+ '--'.join(f'[{v}]' for v in ports)+'\n')
        for load in net.case.loads:
            out.write(f'[{load.source}]--<LOAD {load.id}; customer={load.customer}>--[{load.target}]\n')
        for k,node in enumerate(net.case.nodes):
            out.write(f'[{node.id}] {result.types[k]} V={abs(result.voltage[k]):.9g} pu / {np.angle(result.voltage[k],deg=True):.9g} deg P={actual[k].real:.9g} Q={actual[k].imag:.9g} pu\n')
    selected = np.array([k for k,node in enumerate(net.case.nodes) if focus_feeder is None or node.feeder == focus_feeder],dtype=int)
    if focus_feeder is not None and not len(selected):
        raise ValueError('Selected feeder has no conductors')
    title_suffix = '' if result.converged else ' (NOT CONVERGED)'
    labels = [net.case.nodes[k].id for k in selected]
    svg(directory/'voltage.svg','Conductor voltage magnitude [pu]'+title_suffix,abs(result.voltage[selected]),labels,max_plot_points)
    svg(directory/'angles.svg','Conductor voltage angle [degrees]'+title_suffix,np.angle(result.voltage[selected],deg=True),labels,max_plot_points)
    svg(directory/'convergence.svg','Maximum nonlinear residual'+title_suffix,[r['residual'] for r in result.history],limit=max_plot_points)
    losses = np.sum(result.voltage*np.conj(net.y @ result.voltage))
    summary = {'schema':1,'model':'unbalanced phase-domain steady-state ZIP','converged':result.converged,
               'reason':result.reason,'iterations':len(result.history)-1,'seconds':result.seconds,
               'nodes':len(net.case.nodes),'loads':len(net.case.loads),'y_nnz':net.y.nnz,
               'estimated_peak_gib':net.estimated_bytes/2**30,'three_phase_base_mva':net.case.base_mva,
               'p_loss_mw':float(losses.real*scale),'q_loss_mvar':float(losses.imag*scale),
               'customer_demand_mw':float(load_power.real.sum()*scale),'customer_demand_mvar':float(load_power.imag.sum()*scale),
               'residual':result.history[-1]['residual'],'power_error':result.history[-1]['power_error'],
               'execution':execution,'plot_sample_limit':max_plot_points,'focus_feeder':focus_feeder}
    with (directory/'summary.json').open('w',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False); stream.write('\n')
    shards(directory,'summary',list(summary),[[json.dumps(v) if isinstance(v,dict) else v for v in summary.values()]])
    (directory/'INCOMPLETE').unlink()
    return summary
