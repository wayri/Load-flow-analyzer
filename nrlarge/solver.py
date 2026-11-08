"""Damped sparse rectangular Newton-Raphson, with CPU physical residual checks."""
from dataclasses import dataclass
import time
import numpy as np
from .model import evaluate
from .linear import LocalBackend


@dataclass
class Result:
    voltage: np.ndarray
    types: np.ndarray
    history: list
    converged: bool
    reason: str
    seconds: float


def solve(net, backend=None, tolerance=1e-8, max_iterations=100):
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid nonlinear tolerance/iteration count')
    start = time.perf_counter()
    own_backend = backend is None
    backend = LocalBackend(method='direct') if backend is None else backend
    voltage,types,q = net.initial.copy(),net.types.copy(),net.q.copy()
    history = []
    reason, converged = 'iteration_limit', False
    try:
        for step in range(max_iterations+1):
            f,j,actual,_ = evaluate(net,voltage,types,q)
            error = float(np.max(np.abs(f),initial=0))
            pq = (types == 'PQ')
            power_error = float(np.max(np.abs(actual[pq]-(net.p[pq]+1j*q[pq])),initial=0))
            # Magnitude control rows use V^2, PV P uses pu power, ordinary rows
            # use pu current. Also require explicit PQ complex-power balance.
            if max(error,power_error) < tolerance:
                pv = types == 'PV'
                bounded = np.clip(actual.imag,net.qmin,net.qmax)
                changed = pv & (np.abs(bounded-actual.imag)>tolerance)
                if np.any(changed):
                    types[changed] = 'PQ'; q[changed] = bounded[changed]
                    f,j,actual,_ = evaluate(net,voltage,types,q)
                    error = float(np.max(np.abs(f),initial=0))
                    pq = types == 'PQ'
                    power_error = float(np.max(np.abs(actual[pq]-(net.p[pq]+1j*q[pq])),initial=0))
                else:
                    history.append({'iteration':step,'residual':error,'power_error':power_error})
                    reason,converged = 'converged',True
                    break
            record = {'iteration':step,'residual':error,'power_error':power_error}
            history.append(record)
            if step == max_iterations:
                break
            if not len(f):
                raise ArithmeticError('Empty unknown system with nonzero physical mismatch')
            dx = backend.solve(j,-f).reshape((-1,2))
            record.update(backend.stats)
            accepted = False
            for alpha in 2.0**-np.arange(16):
                trial = voltage.copy()
                trial[net.free] += alpha*(dx[:,0]+1j*dx[:,1])
                if not np.isfinite(trial).all():
                    continue
                try:
                    tf,_,_ = evaluate(net,trial,types,q,jacobian=False)
                except ValueError:
                    continue
                if np.max(np.abs(tf),initial=0) < error:
                    voltage = trial; record['damping'] = float(alpha); accepted = True; break
            if not accepted:
                reason = 'line_search_failed'; break
    finally:
        if own_backend:
            backend.close()
    return Result(voltage,types,history,converged,reason,time.perf_counter()-start)
