"""Reproduce the compact IEEE and contemporary-grid study collection."""
import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import networkx as nx
import numpy as np
from .cases import load_matpower, to_phase_case, independent_reference
from nrlarge.examples import unbalanced_example
from nrlarge.linear import LocalBackend
from nrlarge.model import Load, Network, add_series, evaluate
from nrlarge.reports import write_reports
from nrlarge.solver import solve

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT/'benchmarks/sources/matpower'
MAX_OUTPUT_BYTES = 16*2**20


@dataclass
class Study:
    name: str
    title: str
    case: object
    description: str
    parameters: dict
    dataset: object = None
    expected_converged: bool = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(path):
    """Code hashes survive Git's platform-specific working-tree newlines."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def verify_sources():
    provenance = json.loads((SOURCES/'provenance.json').read_text(encoding='utf-8'))
    for item in provenance['files']:
        path = SOURCES/item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Pinned source changed: {path.name}')
    return provenance


def studies():
    result = []
    for filename,size in [('case14',14),('case_ieee30',30),('case57',57),('case118',118),('case300',300)]:
        data = load_matpower(SOURCES/f'{filename}.m')
        result.append(Study(f'ieee{size}',f'IEEE {size}-bus',to_phase_case(data),
                            'Official MATPOWER-format IEEE case with source generator Q limits.',
                            {'source_file':filename+'.m','q_limits':'aggregate PV generator limits; SLACK Q free'},data))
    stressed = deepcopy(result[0].dataset); stressed.bus[:,2:4] *= 1.5
    result.append(Study('ieee14-load-stress','IEEE 14: demand +50%',to_phase_case(stressed),
                        'Static demand increase with original generator P schedules and voltage setpoints.',
                        {'source_file':'case14.m','demand_multiplier':1.5},stressed))
    outage = deepcopy(result[3].dataset)
    graph = nx.MultiGraph(); graph.add_nodes_from(int(b) for b in outage.bus[:,0])
    for k,row in enumerate(outage.branch):
        if row[10] > 0: graph.add_edge(int(row[0]),int(row[1]),key=k)
    chosen = None
    for k,row in enumerate(outage.branch):
        if row[10] <= 0: continue
        test = graph.copy(); test.remove_edge(int(row[0]),int(row[1]),key=k)
        if nx.is_connected(test): chosen = k; break
    if chosen is None: raise ValueError('No non-islanding outage found')
    outage.branch[chosen,10] = 0
    result.append(Study('ieee118-line-outage','IEEE 118: single-line outage',to_phase_case(outage),
                        'First non-islanding line in source row order is removed; remaining controls are unchanged.',
                        {'source_file':'case118.m','branch_row_1based':chosen+1,
                         'from_bus':int(outage.branch[chosen,0]),'to_bus':int(outage.branch[chosen,1])},outage))
    base = unbalanced_example(); base.base_mva = .1
    for node in base.nodes: node.region = 'illustrative-lv'
    for load in base.loads: load.region = 'illustrative-lv'
    result.append(Study('lv-base','Four-wire LV: base demand',base,
                        'Repository four-wire mutual-impedance/ZIP/delta example, scaled to a 100 kVA three-phase base.',
                        {'base_mva':.1,'source':'nrlarge.examples.unbalanced_example'}))
    ev = deepcopy(base)
    ev.loads.append(Load('EV-A','service.A','service.N',.45,.045,customer='two-EV-chargers',feeder='F1',region='illustrative-lv'))
    result.append(Study('lv-ev-evening','Four-wire LV: EV phase imbalance',ev,
                        'Two illustrative 7.5 kW chargers add 15 kW and 1.5 kvar to phase A.',
                        {'added_phase':'A','added_kw':15,'added_kvar':1.5,'load_model':'constant power'}))
    managed = deepcopy(ev); managed.loads[-1].p *= .5; managed.loads[-1].q *= .5
    result.append(Study('lv-managed-ev','Four-wire LV: managed EV charging',managed,
                        'The additional EV charging demand is capped at half of the evening case.',
                        {'ev_power_fraction':.5,'added_kw':7.5,'added_kvar':.75}))
    rooftop = deepcopy(base)
    for load in rooftop.loads: load.p *= .35; load.q *= .35
    pv_powers = {'A':1.2,'B':.7,'C':.9}
    for phase,power in pv_powers.items():
        rooftop.loads.append(Load('rooftop-'+phase,'service.'+phase,'service.N',-power,0,
                                  customer='rooftop-'+phase,feeder='F1',region='illustrative-lv'))
    result.append(Study('lv-rooftop-export','Four-wire LV: midday rooftop export',rooftop,
                        'Customer demand falls to 35% and unequal fixed-P rooftop generation exports through the feeder.',
                        {'demand_fraction':.35,'phase_generation_kw':{'A':40,'B':70/3,'C':30},'generator_q_pu':0}))
    support = deepcopy(rooftop)
    # This separate control-bound example uses the engine's ideal grounded
    # per-conductor source model, distinct from the neutral-connected PV ports.
    support.loads = [load for load in support.loads if not load.id.startswith('rooftop-')]
    for node in support.nodes:
        if node.id in ('service.A','service.B','service.C'):
            node.type = 'PV'; node.p = pv_powers[node.phase]; node.vm = 1.02; node.qmin = -.03; node.qmax = .03
    result.append(Study('lv-reactive-controls','LV: ideal grounded-source Q limits',support,
                        'Ideal ground-referenced sources target 1.02 pu with +/-1 kvar capability; source current returns to the ground reference.',
                        {'magnitude_setpoint_pu':1.02,'per_phase_q_limit_kvar':1,'controller':'ideal phase-to-ground PV control with Q-bound conversion'}))
    for factor,name in [(2,'lv-neutral-impedance'),(4,'lv-neutral-extreme')]:
        neutral = deepcopy(ev)
        z = np.full((4,4),.002+.015j,dtype=complex); np.fill_diagonal(z,.03+.1j); z[3,3] = factor*(.08+.15j)
        neutral.stamps = []
        add_series(neutral.stamps,'four-wire-line',['source.'+p for p in 'ABCN'],['service.'+p for p in 'ABCN'],z)
        result.append(Study(name,f'Four-wire LV: neutral impedance x{factor}',neutral,
                            f'The EV evening case is repeated with {factor} times the neutral self-impedance.',
                            {'neutral_self_impedance_multiplier':factor,'phase_loads':'same as lv-ev-evening'},expected_converged=factor<4))
    return result


def physical_voltages(case, voltage):
    index = {node.id:k for k,node in enumerate(case.nodes)}
    neutrals = {node.bus:voltage[k] for k,node in enumerate(case.nodes) if node.phase == 'N'}
    selected = [(k,node) for k,node in enumerate(case.nodes) if node.phase in ('A','B','C')]
    magnitudes = np.array([abs(voltage[k]-neutrals.get(node.bus,0j)) for k,node in selected])
    by_bus = {}
    for k,node in selected: by_bus.setdefault(node.bus,{})[node.phase] = voltage[k]-neutrals.get(node.bus,0j)
    unbalance = []
    alpha = np.exp(2j*np.pi/3)
    for bus,phases in by_bus.items():
        if all(p in phases for p in 'ABC'):
            a,b,c = (phases[p] for p in 'ABC')
            positive = (a+alpha*b+alpha**2*c)/3
            negative = (a+alpha**2*b+alpha*c)/3
            if abs(positive)>1e-12: unbalance.append((bus,100*abs(negative)/abs(positive)))
    return selected,magnitudes,neutrals,unbalance


def metrics(net, result):
    _,generation,current = evaluate(net,result.voltage,result.types,jacobian=False)
    demand = (net.a @ result.voltage)*np.conj(current)
    absorption = np.sum(result.voltage*np.conj(net.y @ result.voltage))
    source = np.array([n.type in ('SLACK','PV') or n.p != 0 or n.q != 0 for n in net.case.nodes])
    supplied = generation[source].sum(); scale = net.case.base_mva/3
    _,vm,neutral,unbalance = physical_voltages(net.case,result.voltage)
    slack = np.array([n.type == 'SLACK' for n in net.case.nodes])
    return {'converged':result.converged,'reason':result.reason,'iterations':len(result.history)-1,
            'seconds':result.seconds,'conductors':len(net.case.nodes),'physical_buses':len({n.bus for n in net.case.nodes}),
            'residual':result.history[-1]['residual'],'power_error':result.history[-1]['power_error'],
            'min_phase_to_neutral_vm_pu':float(vm.min()),'max_phase_to_neutral_vm_pu':float(vm.max()),
            'max_neutral_vm_pu':float(max((abs(v) for v in neutral.values()),default=0)),
            'max_negative_sequence_vuf_pct':float(max((v for _,v in unbalance),default=0)),
            'net_explicit_demand_mw':float(demand.real.sum()*scale),
            'gross_load_mw':float(np.maximum(demand.real,0).sum()*scale),
            'embedded_generation_mw':float(-np.minimum(demand.real,0).sum()*scale),
            'net_nodal_source_mw':float(supplied.real*scale),
            'positive_nodal_sources_mw':float(np.maximum(generation.real[source],0).sum()*scale),
            'nodal_source_absorption_mw':float(-np.minimum(generation.real[source],0).sum()*scale),
            'slack_mw':float(generation.real[slack].sum()*scale),'p_loss_mw':float(absorption.real*scale),
            'network_q_absorption_mvar':float(absorption.imag*scale),
            'power_balance_error_mva':float(abs(supplied-demand.sum()-absorption)*scale),
            'pv_to_pq_conductors':int(sum(n.type=='PV' and t=='PQ' for n,t in zip(net.case.nodes,result.types)))}


def csv_write(path, fields, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k:row.get(k,'') for k in fields})


def reference_check(study, result, directory=None):
    ref = independent_reference(study.dataset,tolerance=1e-10)
    by_bus = {str(int(i)):v for i,v in zip(ref.bus_ids,ref.voltage)}
    rows = []
    for k,node in enumerate(study.case.nodes):
        expected = by_bus[node.bus]*np.exp(1j*np.deg2rad({'A':0,'B':-120,'C':120}[node.phase]))
        rows.append({'id':node.id,'reference_re_pu':expected.real,'reference_im_pu':expected.imag,
                     'computed_re_pu':result.voltage[k].real,'computed_im_pu':result.voltage[k].imag,
                     'complex_error_pu':float(abs(result.voltage[k]-expected))})
    maximum = max(row['complex_error_pu'] for row in rows)
    if not result.converged or not ref.converged or maximum > 2e-7:
        raise ArithmeticError(f'{study.name}: phase/polar reference comparison failed; error={maximum}')
    if directory is not None:
        csv_write(directory/'reference-voltage.csv',list(rows[0]),rows)
    return {'method':'separate balanced polar power-balance NR; independent Y-bus/Jacobian',
            'converged':ref.converged,'residual':ref.residual,'iterations':ref.iterations,
            'max_complex_voltage_error_pu':maximum,'acceptance_tolerance_pu':2e-7}


def equipment_checks(study, net, result, directory):
    if study.dataset is None: return {}
    data = study.dataset; index = net.index; scale = data.base_mva/3
    _,generation,_ = evaluate(net,result.voltage,result.types,jacobian=False)
    gen_rows = []
    for bus in data.bus:
        identifier = str(int(bus[0])); gens = data.gen[(data.gen[:,0]==bus[0]) & (data.gen[:,7]>0)]
        if not len(gens): continue
        value = sum(generation[index[identifier+'.'+p]] for p in 'ABC')*scale
        gen_rows.append({'bus':identifier,'p_mw':value.real,'q_mvar':value.imag,'pmin_mw':gens[:,9].sum(),
                         'pmax_mw':gens[:,8].sum(),'qmin_mvar':gens[:,4].sum(),'qmax_mvar':gens[:,3].sum(),
                         'p_within_bounds':bool(gens[:,9].sum()-1e-6<=value.real<=gens[:,8].sum()+1e-6),
                         'q_within_bounds':bool(gens[:,4].sum()-1e-6<=value.imag<=gens[:,3].sum()+1e-6),
                         'control':result.types[index[identifier+'.A']]})
    csv_write(directory/'generator-checks.csv',list(gen_rows[0]),gen_rows)
    powers = {}
    for element,terms in groupby(study.case.stamps,key=lambda s:s[0]):
        if not element.startswith('branch-'): continue
        currents = {}
        for _,a,b,y in terms: currents[a] = currents.get(a,0j)+y*result.voltage[index[b]]
        terminals = {}
        for a,current in currents.items():
            bus = study.case.nodes[index[a]].bus
            terminals[bus] = terminals.get(bus,0j)+result.voltage[index[a]]*np.conj(current)*scale
        powers[element] = terminals
    branch_rows = []
    for k,branch in enumerate(data.branch):
        if branch[10]<=0: continue
        terminals = powers[f'branch-{k+1}']; sf = terminals[str(int(branch[0]))]; st = terminals[str(int(branch[1]))]
        rating = float(branch[5]); loading = 100*max(abs(sf),abs(st))/rating if rating>0 else ''
        branch_rows.append({'branch_row':k+1,'from_bus':int(branch[0]),'to_bus':int(branch[1]),'p_from_mw':sf.real,
                            'q_from_mvar':sf.imag,'p_to_mw':st.real,'q_to_mvar':st.imag,'p_loss_mw':(sf+st).real,
                            'rate_a_mva':rating if rating>0 else '', 'loading_pct':loading})
    csv_write(directory/'branch-checks.csv',list(branch_rows[0]),branch_rows)
    vm_rows = []
    for bus in data.bus:
        vm = abs(result.voltage[index[str(int(bus[0]))+'.A']])
        vm_rows.append({'bus':int(bus[0]),'vm_pu':vm,'vmin_pu':bus[12],'vmax_pu':bus[11],
                        'within_source_bounds':bool(bus[12]-1e-8<=vm<=bus[11]+1e-8)})
    csv_write(directory/'voltage-checks.csv',list(vm_rows[0]),vm_rows)
    return {'voltage_bound_violations':sum(not r['within_source_bounds'] for r in vm_rows),
            'aggregate_generator_p_bound_violations':sum(not r['p_within_bounds'] for r in gen_rows),
            'aggregate_generator_q_bound_violations':sum(not r['q_within_bounds'] for r in gen_rows),
            'rated_branches':sum(r['rate_a_mva']!='' for r in branch_rows),
            'rated_branch_overloads':sum(r['loading_pct']!='' and r['loading_pct']>100 for r in branch_rows)}


def manifest():
    import importlib.metadata
    paths = ['nrlarge/model.py','nrlarge/solver.py','nrlarge/linear.py','nrlarge/reports.py',
             'nrlarge/examples.py','nrlarge/accelerators.py','benchmarks/cases.py',
             'benchmarks/figures.py','benchmarks/run.py','benchmarks/verify.py']
    stamp = datetime.now(timezone.utc)
    return {'date_utc':stamp.date().isoformat(),'generated_at_utc':stamp.isoformat(),
            'python':sys.version.split()[0],'platform':platform.platform(),
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','matplotlib','networkx','Pillow')},
            'source_provenance':verify_sources(),'implementation_sha256':{p:code_sha(ROOT/p) for p in paths},
            'implementation_hash_basis':'file bytes with CRLF normalized to LF; source data/artifacts use exact bytes',
            'solver':'NRlarge sparse rectangular Newton; CPU sparse-LU linear solve',
            'tolerance':1e-10,'max_iterations':100,'q_limit_policy':'PV converted after current system convergence; SLACK Q free'}


def run_case(study, directory=None):
    net = Network(study.case,memory_gib=.5)
    backend = LocalBackend(method='direct')
    try: result = solve(net,backend,tolerance=1e-10,max_iterations=100)
    finally: backend.close()
    if result.converged != study.expected_converged:
        raise ArithmeticError(f'{study.name}: expected converged={study.expected_converged}, received {result.converged} ({result.reason})')
    summary = metrics(net,result)
    record = {'name':study.name,'title':study.title,'description':study.description,
              'parameters':study.parameters,'expected_converged':study.expected_converged,'metrics':summary}
    if study.dataset is not None: record['reference'] = reference_check(study,result)
    if directory is not None:
        write_reports(net,result,directory,{'mode':'local','linear':'sparse direct','threads':1},max_output_mib=8)
        if study.dataset is not None:
            reference_check(study,result,directory)
            record['equipment_checks'] = equipment_checks(study,net,result,directory)
        from .figures import render_case
        record['figures'] = render_case(study.case,result,directory/'figures',study.title)
        (directory/'study.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return record,result


def supplementary(selected, directory):
    """Independent load snapshots and a one-run backend comparison."""
    base = next((s for s in selected if s.name=='ieee14'),None)
    if base is None: return [],[]
    sweep,voltages = [],[]
    for factor in np.round(np.arange(.8,1.81,.1),2):
        data = deepcopy(base.dataset); data.bus[:,2:4] *= factor
        case = to_phase_case(data); net = Network(case,memory_gib=.5)
        result = solve(net,LocalBackend(method='direct'),tolerance=1e-10)
        m = metrics(net,result)
        sweep.append({'demand_multiplier':float(factor),'converged':result.converged,'reason':result.reason,
                      'min_vm_pu':m['min_phase_to_neutral_vm_pu'] if result.converged else '',
                      'max_vm_pu':m['max_phase_to_neutral_vm_pu'] if result.converged else '',
                      'p_loss_mw':m['p_loss_mw'] if result.converged else '',
                      'pv_to_pq_conductors':m['pv_to_pq_conductors'],'residual':m['residual']})
        for k,node in enumerate(case.nodes):
            voltages.append({'demand_multiplier':factor,'id':node.id,'vm_pu':abs(result.voltage[k]),
                             'va_deg':np.angle(result.voltage[k],deg=True),'converged':result.converged,
                             'state':'solution' if result.converged else 'last iterate'})
    csv_write(directory/'load-sweep.csv',list(sweep[0]),sweep)
    csv_write(directory/'load-sweep-voltages.csv',list(voltages[0]),voltages)
    net = Network(base.case,memory_gib=.5)
    reference = solve(net,LocalBackend(method='direct'),tolerance=1e-10)
    rows = []
    for label,method,threads,accelerator in [('CPU sparse LU','direct',1,'cpu'),('CPU GMRES, 1 thread','gmres',1,'cpu'),
                                            ('CPU GMRES, 4 threads','gmres',4,'cpu'),('OpenCL FP64','gmres',1,'opencl'),
                                            ('OpenGL FP64','gmres',1,'opengl')]:
        backend = LocalBackend(threads=threads,method=method,accelerator=accelerator)
        try:
            result = solve(net,backend,tolerance=1e-10)
            error = float(np.max(abs(result.voltage-reference.voltage)))
            if not result.converged or error>2e-7: raise ArithmeticError(f'{label}: backend comparison failed')
            rows.append({'backend':label,'status':'passed','seconds':result.seconds,'iterations':len(result.history)-1,
                         'max_voltage_difference_pu':error,'residual':result.history[-1]['residual'],'detail':''})
        except Exception as error:
            from nrlarge.accelerators import AcceleratorUnavailable
            if not isinstance(error,AcceleratorUnavailable): raise
            rows.append({'backend':label,'status':'unavailable','seconds':'','iterations':'',
                         'max_voltage_difference_pu':'','residual':'','detail':str(error)})
        finally: backend.close()
    csv_write(directory/'backend-comparison.csv',list(rows[0]),rows)
    from nrlarge.accelerators import accelerator_info
    (directory/'hardware.json').write_text(json.dumps(accelerator_info(),indent=2)+'\n',encoding='utf-8')
    return sweep,rows


def overview(records, sweep, backends, directory):
    from .figures import _STYLE,_save
    import matplotlib.pyplot as plt
    files = {}
    with plt.rc_context(_STYLE):
        fig,axes = plt.subplots(1,2,figsize=(11.5,6.4),layout='constrained')
        stable = [r for r in records if r['metrics']['converged']]
        standards = [r for r in stable if r['name'] in ('ieee14','ieee30','ieee57','ieee118','ieee300')]
        axes[0].bar([r['title'] for r in standards],[r['metrics']['p_loss_mw'] for r in standards],color='#1479b8')
        axes[0].tick_params(axis='x',rotation=35);axes[0].set(ylabel='Active network loss [MW]',title='IEEE baseline losses')
        lv = [r for r in stable if r['name'].startswith('lv-')]
        x = np.arange(len(lv))
        axes[1].plot(x,[r['metrics']['min_phase_to_neutral_vm_pu'] for r in lv],'o-',label='Minimum')
        axes[1].plot(x,[r['metrics']['max_phase_to_neutral_vm_pu'] for r in lv],'s-',label='Maximum')
        axes[1].set_xticks(x,[r['name'].replace('lv-','') for r in lv],rotation=35,ha='right')
        axes[1].set(ylabel='Phase-to-neutral voltage [pu]',title='Converged LV snapshots');axes[1].legend()
        for ax in axes: ax.grid(axis='y',alpha=.5)
        fig.suptitle('IEEE benchmarks and contemporary-grid snapshots',weight='bold')
        _save(fig,directory,'comparison',files)
        if sweep:
            fig,axes = plt.subplots(1,2,figsize=(11.5,6.4),layout='constrained')
            good = [r for r in sweep if r['converged']]
            axes[0].plot([r['demand_multiplier'] for r in good],[r['min_vm_pu'] for r in good],'o-',color='#1479b8')
            axes[0].set(xlabel='Demand multiplier',ylabel='Minimum phase voltage [pu]',title='IEEE 14 demand sensitivity')
            axes[1].plot([r['demand_multiplier'] for r in good],[r['p_loss_mw'] for r in good],'o-',color='#df7b1c')
            axes[1].set(xlabel='Demand multiplier',ylabel='Active network loss [MW]',title='IEEE 14 demand sensitivity')
            for ax in axes:ax.grid(alpha=.5)
            fig.suptitle('Independent steady-state load snapshots · PV Q limits enabled',weight='bold')
            _save(fig,directory,'load-sensitivity',files)
        if backends:
            fig,axes = plt.subplots(1,2,figsize=(11.5,6.4),layout='constrained')
            good = [r for r in backends if r['status']=='passed']; labels = [r['backend'] for r in good]
            axes[0].bar(labels,[r['seconds'] for r in good],color='#1479b8');axes[0].set(ylabel='Solve time [seconds]',title='One-run timing, setup included')
            axes[1].bar(labels,[max(r['max_voltage_difference_pu'],1e-16) for r in good],color='#198b66')
            axes[1].set_yscale('log');axes[1].set(ylabel='Maximum complex-voltage difference [pu]',title='Agreement with sparse-LU reference')
            for ax in axes: ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.5)
            fig.suptitle('IEEE 14 · CPU and FP64 GPU paths',weight='bold')
            _save(fig,directory,'backends',files)
    return files


def gallery(records, directory):
    run_date = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))['date_utc']
    lines = ['# IEEE and contemporary-grid results','',
             f'Generated by `python -m benchmarks.run` on {run_date} UTC. Each case includes reloadable inputs, numerical outputs, ASCII connectivity and SVG/PNG figures.','',
             '## Results','',
             '| Case | Status | Min phase V [pu] | Max phase V [pu] | Loss [MW] | Reference error [pu] |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];ref=r.get('reference',{});valid=m['converged']
        lines.append(f"| [{r['title']}]({r['name']}/README.md) | {'Converged' if valid else 'Last iterate · '+m['reason']} | {m['min_phase_to_neutral_vm_pu']:.6f} | {m['max_phase_to_neutral_vm_pu']:.6f} | {m['p_loss_mw']:.6g} | {ref.get('max_complex_voltage_error_pu',float('nan')):.3g} |".replace('| nan |','| — |'))
        case = directory/r['name']
        text = [f"# {r['title']}",'',r['description'],'',f"**Status:** {'converged' if valid else 'solver stopped; displayed values are the final iterate'} ({m['reason']}).",'',
                '## Parameters','', '```json',json.dumps(r['parameters'],indent=2),'```','',
                '## Outputs','',
                '- [Numerical summary](study.json), [CSV summary](summary-00000.csv), [full ASCII network](network.txt).',
                '- [Reloadable inputs](inputs/), [conductor results](nodes-00000.csv), [load results](loads-00000.csv), [element terminals](terminals-00000.csv), [iteration history](convergence-00000.csv).']
        if 'reference' in r:
            text += ['- [Independent polar voltage comparison](reference-voltage.csv), [source voltage-bound checks](voltage-checks.csv), [generator checks](generator-checks.csv), [branch checks](branch-checks.csv).']
        text += ['', '## Figures','']
        for stem,label in [('network','Network diagram'),('voltage','Phase and neutral voltage'),('angles','Voltage angles'),('convergence','Convergence'),('power','Power balance')]:
            text += [f'### {label}','',f'![{label}](figures/{stem}.png)','',f'[Zoomable SVG](figures/{stem}.svg)','']
        text += ['[Collection index](../README.md) · [Acknowledgements and disclaimer](../../../ACKNOWLEDGEMENTS.md)','']
        (case/'README.md').write_text('\n'.join(text),encoding='utf-8')
    lines += ['', '![Collection comparison](comparison.png)','', '[Zoomable comparison](comparison.svg)','',
              '## Demand sensitivity and compute paths','',
              '![IEEE 14 load sensitivity](load-sensitivity.png)','',
              '[Load-sweep CSV](load-sweep.csv) · [All sweep conductor voltages](load-sweep-voltages.csv) · [SVG](load-sensitivity.svg)','',
              '![Compute-path comparison](backends.png)','',
              '[Backend comparison CSV](backend-comparison.csv) · [Hardware receipt](hardware.json) · [SVG](backends.svg)','',
              '## Data and numerical conventions','',
              'MATPOWER provides the pinned case data. The separate balanced polar implementation in `benchmarks/cases.py` supplies the comparison values. The phase engine uses a per-conductor power base equal to the three-phase case MVA base divided by three. PV source Q bounds are enabled; SLACK Q remains unconstrained.','',
              'Voltage, generator and branch check files report source-data limits. Power-flow controls enforce PV Q bounds; generator P bounds, SLACK Q and branch thermal ratings are recorded for assessment. Blank branch ratings indicate unrated source entries.','',
              'LV voltage metrics use phase-to-neutral voltage. Rooftop generation uses negative constant-power loads between each phase and the actual service neutral. The separate Q-control example uses ideal ground-referenced per-conductor sources and records that return path in its parameters. Negative SLACK MW denotes feeder export. Negative-sequence voltage unbalance is 100 × |V2| / |V1| at buses containing A/B/C phases.','',
              'The extreme neutral case records a nonconverged solve. Its CSV rows and plots carry that status, and its final iterate is excluded from the converged-LV comparison plot. Demand-sweep points are independent steady-state snapshots.','',
              'Modern scenario themes follow distribution-grid studies such as [NIST TN 2289](https://www.nist.gov/publications/impact-price-responsive-load-and-volt-var-power-quality-losses-and-customer-economics) and [PNNL-22064](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-22064.pdf). Exact selected settings are in each study receipt and its input CSVs.','',
              'Network coordinates are deterministic force-layout coordinates. Full conductor connectivity is retained in each ASCII diagram and input stamps. The node drawings group phase conductors by physical bus.','',
              '[Source provenance](../sources/matpower/provenance.json) · [Run manifest](manifest.json) · [Machine-readable index](index.json) · [CSV comparison](comparison.csv) · [Artifact hashes](artifacts.json) · [Acknowledgements and disclaimer](../../ACKNOWLEDGEMENTS.md)','']
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'benchmarks/results')
    parser.add_argument('--only',help='Comma-separated study identifiers')
    parser.add_argument('--check-only',action='store_true',help='Numerical checks in RAM without exports')
    args = parser.parse_args(argv)
    verify_sources(); selected = studies()
    if args.only:
        names = set(args.only.split(',')); selected = [s for s in selected if s.name in names]
        if names != {s.name for s in selected}: raise ValueError('Unknown study identifier')
    if not args.check_only:
        if args.out.exists() and any(args.out.iterdir()): raise ValueError('Choose a new/empty collection output directory')
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'manifest.json').write_text(json.dumps(manifest(),indent=2)+'\n',encoding='utf-8')
    records = []
    for study in selected:
        record,_ = run_case(study,None if args.check_only else args.out/study.name); records.append(record)
        m=record['metrics']
        print(json.dumps({'case':study.name,'converged':m['converged'],'iterations':m['iterations'],
                          'min_vm_pu':m['min_phase_to_neutral_vm_pu'],'max_vm_pu':m['max_phase_to_neutral_vm_pu'],
                          'reference_error_pu':record.get('reference',{}).get('max_complex_voltage_error_pu')},allow_nan=False),flush=True)
        if not args.check_only:
            total = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
            if total>MAX_OUTPUT_BYTES: raise OSError('Collection exceeded its 16 MiB publication budget')
    if not args.check_only:
        (args.out/'index.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        fields = ['name','title',*records[0]['metrics']]
        csv_write(args.out/'comparison.csv',fields,({'name':r['name'],'title':r['title'],**r['metrics']} for r in records))
        sweep,backends = supplementary(selected,args.out)
        overview(records,sweep,backends,args.out)
        if sweep: gallery(records,args.out)
        files = [{'path':p.relative_to(args.out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(args.out.rglob('*')) if p.is_file() and p.name!='artifacts.json']
        (args.out/'artifacts.json').write_text(json.dumps({'files':files},indent=2)+'\n',encoding='utf-8')
        if sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())>MAX_OUTPUT_BYTES:
            raise OSError('Collection exceeded its 16 MiB publication budget')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
