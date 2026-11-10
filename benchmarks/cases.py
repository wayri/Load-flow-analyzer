"""Safe static MATPOWER v2 import and an independent balanced polar reference.

Only literal data assignments are accepted. No MATLAB/Octave code is executed.
The reference deliberately does not call NRlarge stamping, residual or solver code.
Generator limits are aggregated per bus; PV limits are enforced after convergence,
with permanently clamped PQ buses for the remainder of a run. Slack Q is free.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_NUMERIC = re.compile(_NUMBER)


@dataclass
class Dataset:
    name: str
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray


@dataclass
class ReferenceResult:
    bus_ids: np.ndarray
    voltage: np.ndarray
    types: np.ndarray
    generation_power: np.ndarray
    qmin: np.ndarray
    qmax: np.ndarray
    converged: bool
    residual: float
    iterations: int
    reason: str

    @property
    def generator_q(self):
        """Aggregate bus generator Q, per unit on the three-phase case base."""
        return self.generation_power.imag


def _without_comments(text):
    result = []; k = 0; quoted = False
    while k < len(text):
        ch = text[k]
        if ch == "'":
            result.append(ch)
            if quoted and k+1 < len(text) and text[k+1] == "'":
                result.append("'"); k += 2; continue
            quoted = not quoted
        elif ch == '%' and not quoted:
            end = text.find('\n', k)
            k = len(text) if end < 0 else end
            continue
        else:
            result.append(ch)
        k += 1
    if quoted:
        raise ValueError('Unterminated MATLAB string literal')
    return ''.join(result)


def _matrix(text, field):
    rows = []
    for row in re.split(r';|\r?\n', text):
        if not row.strip(): continue
        values = re.split(r'[\s,]+', row.strip())
        if any(_NUMERIC.fullmatch(v) is None for v in values):
            raise ValueError(f'Only finite numeric literals are supported in mpc.{field}')
        rows.append([float(v.replace('D', 'E').replace('d', 'e')) for v in values])
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f'Empty or ragged mpc.{field} matrix')
    value = np.asarray(rows, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f'Non-finite mpc.{field} data')
    return value


def load_matpower(path):
    """Read the documented static subset used by the pinned standard cases.

    Recognized assignments: version, baseMVA, bus, gen, branch, gencost, areas,
    and bus_name (a literal string cell array). Other statements fail explicitly.
    """
    path = Path(path)
    if path.stat().st_size > 2*1024*1024:
        raise ValueError('Static MATPOWER source exceeds the 2 MiB parser bound')
    text = _without_comments(path.read_text(encoding='utf-8-sig'))
    function = re.match(r'\s*function[ \t]+mpc[ \t]*=[ \t]*([A-Za-z]\w*)[ \t]*;?[ \t]*(?:\r?\n|$)', text)
    if function is None:
        raise ValueError('Expected a static function mpc = case_name header')
    name = function[1]; position = function.end(); fields = {}
    while position < len(text):
        whitespace = re.match(r'\s*', text[position:]); position += whitespace.end()
        if position == len(text): break
        if re.fullmatch(r'end\s*;?\s*', text[position:]): break
        assignment = re.match(r'mpc\.([A-Za-z]\w*)\s*=\s*', text[position:])
        if assignment is None:
            raise ValueError('Unsupported executable MATLAB statement: '+text[position:position+70].strip())
        field = assignment[1]; position += assignment.end()
        if field in fields: raise ValueError(f'Duplicate mpc.{field} assignment')
        if field in ('bus','gen','branch','gencost','areas'):
            if position >= len(text) or text[position] != '[':
                raise ValueError(f'mpc.{field} must be a literal numeric matrix')
            end = text.find(']', position+1)
            if end < 0: raise ValueError(f'Unterminated mpc.{field} matrix')
            fields[field] = _matrix(text[position+1:end], field); position = end+1
        elif field == 'bus_name':
            literal = re.match(r"\{(?:\s*'(?:[^']|'')*'\s*;?)*\s*\}", text[position:])
            if literal is None: raise ValueError('mpc.bus_name must be literal strings')
            fields[field] = True; position += literal.end()
        elif field == 'version':
            literal = re.match(r"'2'", text[position:])
            if literal is None: raise ValueError('Only MATPOWER version 2 is supported')
            fields[field] = 2; position += literal.end()
        elif field == 'baseMVA':
            literal = _NUMERIC.match(text, position)
            if literal is None: raise ValueError('baseMVA must be a numeric literal')
            fields[field] = float(literal[0].replace('D','E').replace('d','e')); position = literal.end()
        else:
            raise ValueError(f'Unsupported mpc.{field} assignment')
        terminator = re.match(r'\s*;', text[position:])
        if terminator is None: raise ValueError(f'Expected semicolon after mpc.{field}')
        position += terminator.end()
    if any(key not in fields for key in ('version','baseMVA','bus','gen','branch')):
        raise ValueError('Missing MATPOWER version/baseMVA/bus/gen/branch data')
    dataset = Dataset(name, fields['baseMVA'], fields['bus'], fields['gen'], fields['branch'])
    _validate(dataset)
    return dataset


def _validate(data):
    if not math.isfinite(data.base_mva) or data.base_mva <= 0:
        raise ValueError('baseMVA must be positive and finite')
    for name, minimum in (('bus',13),('gen',10),('branch',13)):
        a = getattr(data,name)
        if not isinstance(a,np.ndarray) or a.ndim != 2 or a.shape[1] < minimum or not np.isfinite(a).all():
            raise ValueError(f'Invalid MATPOWER {name} shape or data')
    bus, gen, branch = data.bus, data.gen, data.branch
    if not len(bus) or len(set(bus[:,0])) != len(bus) or np.any(bus[:,0] != np.floor(bus[:,0])) or np.any(bus[:,0] <= 0):
        raise ValueError('Bus IDs must be positive unique integers')
    ids = set(bus[:,0])
    if np.any(~np.isin(bus[:,1],[1,2,3,4])) or np.any(bus[:,7] <= 0):
        raise ValueError('Unsupported bus type or nonpositive voltage magnitude')
    if any(row[0] not in ids for row in gen) or any(row[0] not in ids or row[1] not in ids or row[0] == row[1] for row in branch):
        raise ValueError('Unknown generator/branch bus or self branch')
    for row in gen:
        if row[7] > 0 and (row[3] < row[4] or row[5] <= 0):
            raise ValueError('Invalid online generator Q limits or voltage setpoint')
    for row in branch:
        if row[10] > 0 and (abs(complex(row[2],row[3])) <= 1e-15 or row[8] < 0):
            raise ValueError('Zero-impedance branch or negative transformer tap unsupported')


def _bus_data(data):
    """Aggregate sources, retaining the original active-bus row order."""
    _validate(data)
    bus = data.bus[data.bus[:,1] != 4].copy()
    if not len(bus): raise ValueError('No active buses')
    ids = bus[:,0].astype(np.int64); index = {int(v):k for k,v in enumerate(ids)}
    pg = np.zeros(len(bus)); qg = pg.copy(); qmin = pg.copy(); qmax = pg.copy()
    vm = bus[:,7].copy(); types = np.array([{1:'PQ',2:'PV',3:'SLACK'}[int(v)] for v in bus[:,1]])
    for k, identifier in enumerate(ids):
        online = data.gen[(data.gen[:,0] == identifier) & (data.gen[:,7] > 0)]
        if len(online):
            pg[k],qg[k],qmax[k],qmin[k] = online[:,1:5].sum(axis=0)/data.base_mva
            if types[k] in ('PV','SLACK'):
                if not np.allclose(online[:,5],online[0,5],rtol=0,atol=1e-10):
                    raise ValueError(f'Conflicting online generator voltage setpoints at bus {identifier}')
                vm[k] = online[0,5]
        elif types[k] == 'SLACK':
            raise ValueError(f'Slack bus {identifier} has no online generator')
        elif types[k] == 'PV': types[k] = 'PQ'
    if not np.any(types == 'SLACK'): raise ValueError('No active slack bus')
    return bus,ids,index,pg,qg,qmin,qmax,vm,types


def to_phase_case(data):
    """Expand a balanced case into independent A/B/C conductors and explicit loads.

    Numerical pu injections/impedances are unchanged on each phase; physical
    conductor power is pu * baseMVA/3. Static shunts remain admittance stamps.
    """
    from nrlarge.model import Case, Node, Load, add_series
    bus,ids,index,pg,qg,qmin,qmax,vm,types = _bus_data(data)
    nodes,loads,stamps = [],[],[]
    for k,row in enumerate(bus):
        identifier = str(int(ids[k])); region = f'area-{int(row[6])}'
        for phase,offset in zip('ABC',(0,-120,120)):
            conductor = identifier+'.'+phase
            nodes.append(Node(conductor,identifier,phase,types[k],
                              0 if types[k] == 'SLACK' else pg[k],
                              qg[k] if types[k] == 'PQ' else 0,vm[k],row[8]+offset,
                              qmin[k],qmax[k],region=region))
            if row[2] != 0 or row[3] != 0:
                loads.append(Load('load-'+conductor,conductor,p=row[2]/data.base_mva,q=row[3]/data.base_mva,
                                  customer=identifier,region=region))
        shunt = complex(row[4],row[5])/data.base_mva
        if shunt != 0:
            for phase in 'ABC': stamps.append(('shunt-'+identifier,identifier+'.'+phase,identifier+'.'+phase,shunt))
    for k,row in enumerate(data.branch):
        a,b = int(row[0]),int(row[1])
        if row[10] <= 0 or a not in index or b not in index: continue
        tap = (row[8] if row[8] != 0 else 1)*np.exp(1j*np.deg2rad(row[9]))
        add_series(stamps,f'branch-{k+1}',[f'{a}.{p}' for p in 'ABC'],[f'{b}.{p}' for p in 'ABC'],
                   np.eye(3)*complex(row[2],row[3]),np.eye(3)*1j*row[4],[tap]*3)
    return Case(nodes,stamps,loads,data.base_mva)


def _reference_admittance(data,bus,index):
    """Separate balanced pi-branch stamp, with no NRlarge stamp helper calls."""
    n = len(bus); y = sparse.lil_matrix((n,n),dtype=complex)
    for row in data.branch:
        if row[10] <= 0 or int(row[0]) not in index or int(row[1]) not in index: continue
        a,b = index[int(row[0])],index[int(row[1])]
        series = 1/complex(row[2],row[3]); charging = 1j*row[4]/2
        ratio = row[8] if row[8] else 1
        tap = ratio*(math.cos(math.radians(row[9]))+1j*math.sin(math.radians(row[9])))
        y[a,a] += (series+charging)/(ratio*ratio)
        y[a,b] -= series/tap.conjugate(); y[b,a] -= series/tap
        y[b,b] += series+charging
    y.setdiag(y.diagonal()+(bus[:,4]+1j*bus[:,5])/data.base_mva)
    return y.tocsr()


def independent_reference(data,tolerance=1e-10,max_iterations=100,enforce_q_limits=True):
    """Damped full-Jacobian polar Newton reference, CPU sparse linear solves.

    P/Q mismatch and generator Q limits use pu on baseMVA. This is an independent
    formulation check, not an assertion that MATPOWER itself was executed.
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations,int) or max_iterations < 1:
        raise ValueError('Invalid reference tolerance/iteration limit')
    bus,ids,index,pg,qg,qmin,qmax,vm,types = _bus_data(data)
    y = _reference_admittance(data,bus,index)
    graph = y.copy(); graph.data = np.ones(graph.nnz)
    _, components = connected_components(graph,directed=False)
    referenced = set(components[types == 'SLACK'])
    if any(v not in referenced for v in components): raise ValueError('Unreferenced electrical island')
    voltage = vm*np.exp(1j*np.deg2rad(bus[:,8])); demand = (bus[:,2]+1j*bus[:,3])/data.base_mva
    specified = pg+1j*qg-demand
    reason,converged = 'iteration_limit',False
    for iteration in range(max_iterations+1):
        current = y @ voltage; power = voltage*np.conj(current)
        angles = np.flatnonzero(types != 'SLACK'); pq = np.flatnonzero(types == 'PQ')
        def mismatch(v):
            s = v*np.conj(y @ v)
            return np.r_[specified.real[angles]-s.real[angles],specified.imag[pq]-s.imag[pq]]
        f = mismatch(voltage); residual = float(np.max(np.abs(f),initial=0))
        if residual < tolerance:
            generation = power+demand
            changed = (types == 'PV') & ((generation.imag < qmin-tolerance) | (generation.imag > qmax+tolerance)) if enforce_q_limits else np.zeros(len(bus),dtype=bool)
            if np.any(changed):
                bounded = np.clip(generation.imag,qmin,qmax)
                types[changed] = 'PQ'
                specified.imag[changed] = bounded[changed]-demand.imag[changed]
                pq = np.flatnonzero(types == 'PQ'); f = mismatch(voltage)
                residual = float(np.max(np.abs(f),initial=0))
            else:
                converged,reason = True,'converged'; break
        if iteration == max_iterations: break
        magnitude = np.abs(voltage); unit = voltage/magnitude
        dv = sparse.diags(voltage); di = sparse.diags(current)
        da = 1j*dv @ (di-y @ dv).conjugate()
        dm = dv @ (y @ sparse.diags(unit)).conjugate()+di.conjugate() @ sparse.diags(unit)
        j = sparse.bmat([[da[angles,:][:,angles].real,dm[angles,:][:,pq].real],
                         [da[pq,:][:,angles].imag,dm[pq,:][:,pq].imag]],format='csc')
        delta = spsolve(j,f)
        if not np.isfinite(delta).all(): reason = 'singular_jacobian'; break
        accepted = False
        for alpha in 2.0**-np.arange(16):
            trial_magnitude = magnitude.copy(); trial_angle = np.angle(voltage)
            trial_angle[angles] += alpha*delta[:len(angles)]
            trial_magnitude[pq] += alpha*delta[len(angles):]
            if np.any(trial_magnitude <= 1e-10): continue
            trial = trial_magnitude*np.exp(1j*trial_angle)
            if np.max(np.abs(mismatch(trial)),initial=0) < residual:
                voltage = trial; accepted = True; break
        if not accepted: reason = 'line_search_failed'; break
    generation = voltage*np.conj(y @ voltage)+demand
    return ReferenceResult(ids,voltage,types,generation,qmin,qmax,converged,residual,iteration,reason)
