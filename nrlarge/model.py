"""Explicit conductor network, general element stamps and wye/delta ZIP loads."""
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee


@dataclass
class Node:
    id: str
    bus: str = ""
    phase: str = "A"
    type: str = "PQ"
    p: float = 0.0
    q: float = 0.0
    vm: float = 1.0
    va: float = 0.0
    qmin: float = -math.inf
    qmax: float = math.inf
    region: str = ""
    feeder: str = ""
    customer: str = ""


@dataclass
class Load:
    id: str
    source: str
    target: str = "GROUND"
    p: float = 0.0
    q: float = 0.0
    z: float = 0.0
    i: float = 0.0
    power: float = 1.0
    nominal: float = 1.0
    customer: str = ""
    feeder: str = ""
    region: str = ""


@dataclass
class Case:
    nodes: list[Node]
    # (element ID, row conductor ID, column conductor ID, complex admittance)
    stamps: list[tuple[str, str, str, complex]]
    loads: list[Load] = field(default_factory=list)
    base_mva: float = 100.0  # THREE-PHASE base; each conductor base is base_mva / 3.


NODE_FIELDS = "id,bus,phase,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu,region,feeder,customer".split(",")
STAMP_FIELDS = "element,row,col,g_pu,b_pu".split(",")
LOAD_FIELDS = "id,from,to,p_pu,q_pu,z_fraction,i_fraction,p_fraction,nominal_vm_pu,customer,feeder,region".split(",")


def records(path, fields):
    """Stream records; do not retain an intermediate copy of the entire CSV."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path}: expected CSV header {','.join(fields)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: incorrect field count")
            if any(len(v) > 512 for v in row.values()):
                raise ValueError(f'{path}:{line}: CSV field exceeds 512 characters')
            yield {k: v.strip() for k, v in row.items()}


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number: {value}")
    return result


def read_case(directory, base_mva=None, memory_gib=4, restart=30):
    directory = Path(directory)
    if base_mva is None:
        manifest = directory / 'case.json'
        if manifest.exists():
            if manifest.stat().st_size > 4096:
                raise ValueError('Case manifest exceeds 4 KiB')
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            if metadata.get('schema') != 1:
                raise ValueError('Unsupported case schema')
            base_mva = metadata['base_mva']
        else:
            base_mva = 100.0
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Invalid ingestion memory budget')
    used = 0
    def reserve(cost):
        nonlocal used
        used += cost
        if used > memory_gib*2**30:
            raise MemoryError('CSV ingestion estimate exceeds configured RAM budget')
    nodes = []
    for r in records(directory / "nodes.csv", NODE_FIELDS):
        reserve(2500+16*(restart+10))
        nodes.append(Node(r['id'], r['bus'], r['phase'], r['type'],
                          *[number(r[k]) for k in ('p_pu','q_pu','vm_pu','va_deg')],
                          number(r['qmin_pu']) if r['qmin_pu'] else -math.inf,
                          number(r['qmax_pu']) if r['qmax_pu'] else math.inf,
                          r['region'], r['feeder'], r['customer']))
    canonical = {v.id:v.id for v in nodes}
    if len(canonical) != len(nodes) or '' in canonical:
        raise ValueError('Conductor IDs must be nonempty and unique')
    stamps = []; element = None
    for r in records(directory / "stamps.csv", STAMP_FIELDS):
        reserve(800)
        if r['row'] not in canonical or r['col'] not in canonical:
            raise ValueError('Unknown conductor in admittance stamp')
        if element != r['element']: element = r['element']
        stamps.append((element, canonical[r['row']], canonical[r['col']], complex(number(r['g_pu']), number(r['b_pu']))))
    loads = []
    if (directory / "loads.csv").exists():
        for r in records(directory / "loads.csv", LOAD_FIELDS):
            reserve(1800)
            loads.append(Load(r['id'], r['from'], r['to'],
                              *[number(r[k]) for k in ('p_pu','q_pu','z_fraction','i_fraction','p_fraction','nominal_vm_pu')],
                              r['customer'], r['feeder'], r['region']))
    return Case(nodes, stamps, loads, base_mva)


def write_case(case, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'case.json').write_text(json.dumps({'schema':1,'base_mva':case.base_mva},allow_nan=False)+'\n',encoding='utf-8')
    def write(name, fields, rows):
        with (directory / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)
    write('nodes.csv', NODE_FIELDS, ([n.id,n.bus,n.phase,n.type,n.p,n.q,n.vm,n.va,
          n.qmin if math.isfinite(n.qmin) else '',n.qmax if math.isfinite(n.qmax) else '',
          n.region,n.feeder,n.customer] for n in case.nodes))
    write('stamps.csv', STAMP_FIELDS, ([e,a,b,y.real,y.imag] for e,a,b,y in case.stamps))
    write('loads.csv', LOAD_FIELDS, ([v.id,v.source,v.target,v.p,v.q,v.z,v.i,v.power,v.nominal,
          v.customer,v.feeder,v.region] for v in case.loads))


def add_series(stamps, element, sources, targets, impedance, charging=None, taps=None):
    """Stamp an invertible coupled impedance matrix; complex from-side taps.

    This utility models a series multi-conductor branch, not every winding group.
    More general transformers can be represented by explicit terminal stamps.
    """
    z = np.asarray(impedance, dtype=np.complex128)
    m = len(sources)
    if len(targets) != m or z.shape != (m, m) or not np.isfinite(z).all():
        raise ValueError('Invalid coupled impedance dimensions/data')
    y = np.linalg.inv(z)
    shunt = np.zeros_like(y) if charging is None else np.asarray(charging, dtype=complex) / 2
    tap = np.eye(m, dtype=complex) if taps is None else np.diag(1 / np.asarray(taps, dtype=complex))
    block = np.block([[tap.conj().T @ (y+shunt) @ tap, -tap.conj().T @ y],
                      [-y @ tap, y+shunt]])
    ids = list(sources) + list(targets)
    for a in range(2*m):
        for b in range(2*m):
            if block[a,b] != 0:
                stamps.append((element, ids[a], ids[b], complex(block[a,b])))


class Network:
    def __init__(self, case, memory_gib=4, restart=30, ordering='rcm'):
        self.case = case
        n = len(case.nodes)
        if not n or not math.isfinite(case.base_mva) or case.base_mva <= 0:
            raise ValueError('Require nodes and a positive finite three-phase MVA base')
        if restart < 2 or not math.isfinite(memory_gib) or memory_gib <= 0:
            raise ValueError('Invalid memory limit or Krylov restart')
        # Conservative preflight includes Python metadata, assembly temporaries,
        # duplicated sparse row shards and GMRES basis; LU fill remains data dependent.
        self.estimated_bytes = n * (2500 + 16*(restart+10)) + len(case.stamps)*800 + len(case.loads)*1800
        if self.estimated_bytes > memory_gib * 2**30:
            raise MemoryError(f'Estimated peak {self.estimated_bytes/2**30:.3f} GiB exceeds limit {memory_gib} GiB')
        self.index = {v.id: i for i,v in enumerate(case.nodes)}
        if len(self.index) != n or any(not v.id or v.id == 'GROUND' for v in case.nodes):
            raise ValueError('Conductor IDs must be nonempty, unique and not GROUND')
        for v in case.nodes:
            if v.type not in ('SLACK','PQ','PV') or v.phase not in ('A','B','C','N','S1','S2'):
                raise ValueError(f'Invalid node type/phase: {v.id}')
            if not all(math.isfinite(x) for x in (v.p,v.q,v.vm,v.va)) or v.vm < 0:
                raise ValueError(f'Invalid node parameters: {v.id}')
            if (v.type == 'PV' or v.p != 0 or v.q != 0) and v.vm <= 1e-10:
                raise ValueError(f'Nonzero voltage required: {v.id}')
            if math.isnan(v.qmin) or math.isnan(v.qmax) or v.qmin > v.qmax or v.qmin == math.inf or v.qmax == -math.inf:
                raise ValueError(f'Invalid reactive limits: {v.id}')
        rr, cc, vv = [], [], []
        previous = None; closed_elements = set()
        for element,a,b,y in case.stamps:
            if element != previous:
                if element in closed_elements:
                    raise ValueError('Group stamp rows contiguously by element for streamed reports')
                if previous is not None: closed_elements.add(previous)
                previous = element
            if not element or a not in self.index or b not in self.index or not np.isfinite(y):
                raise ValueError(f'Invalid admittance stamp: {element}')
            rr.append(self.index[a]); cc.append(self.index[b]); vv.append(y)
        self.y = sparse.coo_matrix((vv,(rr,cc)), shape=(n,n), dtype=complex).tocsr()
        self.y.eliminate_zeros(); self.y.sort_indices()
        lr, lc, lv = [], [], []
        load_ids = set()
        for k,load in enumerate(case.loads):
            if not load.id or load.id in load_ids or load.source not in self.index or (load.target != 'GROUND' and load.target not in self.index) or load.source == load.target:
                raise ValueError(f'Invalid load endpoints/ID: {load.id}')
            load_ids.add(load.id)
            if not all(math.isfinite(v) for v in (load.p,load.q,load.z,load.i,load.power,load.nominal)) or min(load.z,load.i,load.power) < 0 or not math.isclose(load.z+load.i+load.power,1,abs_tol=1e-12) or load.nominal <= 0:
                raise ValueError(f'Invalid ZIP model: {load.id}')
            lr.append(k); lc.append(self.index[load.source]); lv.append(1)
            if load.target != 'GROUND':
                lr.append(k); lc.append(self.index[load.target]); lv.append(-1)
        self.a = sparse.coo_matrix((lv,(lr,lc)),shape=(len(case.loads),n),dtype=float).tocsr()
        graph = self.y.copy(); graph.data = np.ones(graph.nnz)
        unsigned_incidence = abs(self.a)
        graph = graph + unsigned_incidence.T @ unsigned_incidence
        _, components = connected_components(graph, directed=False)
        slack_components = {components[i] for i,v in enumerate(case.nodes) if v.type == 'SLACK'}
        if any(v not in slack_components for v in components):
            raise ValueError('Every electrical component must have a SLACK reference')
        self.initial = np.array([v.vm*np.exp(1j*np.deg2rad(v.va)) for v in case.nodes])
        self.types = np.array([v.type for v in case.nodes])
        self.p = np.array([v.p for v in case.nodes]); self.q = np.array([v.q for v in case.nodes])
        self.vset = np.array([v.vm for v in case.nodes])
        self.qmin = np.array([v.qmin for v in case.nodes]); self.qmax = np.array([v.qmax for v in case.nodes])
        self.free = np.flatnonzero(self.types != 'SLACK')
        if ordering not in ('input','rcm'):
            raise ValueError('Unknown conductor partition ordering')
        self.ordering = ordering
        if ordering == 'rcm' and len(self.free):
            adjacency = graph[self.free,:][:,self.free].tocsr()
            self.free = self.free[reverse_cuthill_mckee(adjacency,symmetric_mode=False)]
        self.power = np.array([complex(v.p,v.q) for v in case.loads])
        self.z = np.array([v.z for v in case.loads]); self.i = np.array([v.i for v in case.loads])
        self.cp = np.array([v.power for v in case.loads]); self.nominal = np.array([v.nominal for v in case.loads])


def current_and_derivatives(voltage, power, z, i, cp, nominal):
    """ZIP current from source to target and analytic rectangular derivatives."""
    r = np.abs(voltage)
    active = np.abs(power) != 0
    if np.any(active & (r < 1e-10) & ((i != 0) | (cp != 0))):
        raise ValueError('Constant current/power device has collapsed terminal voltage')
    safe = np.maximum(r, 1e-10)
    s = np.conj(power)
    f = z/nominal**2 + i/(nominal*safe) + cp/safe**2
    a = i/(nominal*safe**3) + 2*cp/safe**4
    result = s*voltage*f
    de = s*(f-voltage*voltage.real*a)
    df = s*(1j*f-voltage*voltage.imag*a)
    return result, de, df


def evaluate(net, voltage, types=None, q=None, jacobian=True):
    types = net.types if types is None else types
    q = net.q if q is None else q
    load_i, load_e, load_f = current_and_derivatives(net.a @ voltage, net.power, net.z, net.i, net.cp, net.nominal)
    demand_i = net.a.T @ load_i
    physical_i = net.y @ voltage + demand_i
    pv = types == 'PV'; normal = ~pv
    source_power = np.where(normal, net.p+1j*q, 0)
    source_i, source_e, source_f = current_and_derivatives(voltage, source_power, 0, 0, 1, 1)
    residual = physical_i-source_i
    actual = voltage*np.conj(physical_i)
    residual.real[pv] = actual.real[pv]-net.p[pv]
    residual.imag[pv] = np.abs(voltage[pv])**2-net.vset[pv]**2
    f = np.column_stack((residual.real[net.free], residual.imag[net.free])).ravel()
    if not np.isfinite(f).all():
        raise ArithmeticError('Non-finite residual')
    if not jacobian:
        return f, actual, load_i
    h = net.y + net.a.T @ sparse.diags(load_e) @ net.a - sparse.diags(source_e)
    k = 1j*net.y + net.a.T @ sparse.diags(load_f) @ net.a - sparse.diags(source_f)
    ordinary = sparse.diags(normal.astype(float)); controlled = sparse.diags(pv.astype(float))
    vconj = sparse.diags(np.conj(voltage))
    he = ordinary @ h.real + controlled @ (vconj @ h).real + sparse.diags(pv*physical_i.real)
    hf = ordinary @ k.real + controlled @ (vconj @ k).real + sparse.diags(pv*physical_i.imag)
    qe = ordinary @ h.imag + sparse.diags(pv*2*voltage.real)
    qf = ordinary @ k.imag + sparse.diags(pv*2*voltage.imag)
    rows = np.column_stack((net.free, len(voltage)+net.free)).ravel()
    j = sparse.bmat([[he,hf],[qe,qf]],format='csr')[rows,:][:,rows].tocsr()
    j.eliminate_zeros(); j.sort_indices()
    if not np.isfinite(j.data).all():
        raise ArithmeticError('Non-finite Jacobian')
    return f, j, actual, load_i
