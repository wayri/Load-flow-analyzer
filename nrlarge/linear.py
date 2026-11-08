"""Sparse direct reference and threaded/block-distributed Krylov linear algebra."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu


def partitions(n, count):
    if n % 2 or count < 1:
        raise ValueError('Expected paired real/imaginary unknowns and positive workers')
    count = min(count, max(1,n//2))
    return [(2*((n//2)*k//count),2*((n//2)*(k+1)//count)) for k in range(count)]


class RowShard:
    def __init__(self, matrix, diagonal, accelerator='cpu', fill_factor=4, threads=1):
        if not 1 <= threads <= 256:
            raise ValueError('Worker CPU threads must be in 1..256')
        self.matrix = sparse.csr_matrix(matrix, dtype=np.float64)
        if not np.isfinite(self.matrix.data).all():
            raise ValueError('Non-finite sparse matrix')
        self.factor = spilu(sparse.csc_matrix(diagonal), drop_tol=1e-5, fill_factor=fill_factor)
        self.accelerator = None
        self.pool = None
        self.chunks = []
        if accelerator != 'cpu':
            from .accelerators import create_accelerator
            self.accelerator = create_accelerator(accelerator,self.matrix)
        elif threads > 1:
            count = min(threads,self.matrix.shape[0])
            self.chunks = [self.matrix[self.matrix.shape[0]*k//count:self.matrix.shape[0]*(k+1)//count,:] for k in range(count)]
            self.pool = ThreadPoolExecutor(max_workers=count)

    def matvec(self, x):
        if self.accelerator is not None:
            result = self.accelerator.matvec(x)
        elif self.pool is not None:
            result = np.concatenate(list(self.pool.map(lambda chunk: chunk @ x,self.chunks)))
        else:
            result = self.matrix @ x
        if not np.isfinite(result).all():
            raise ArithmeticError('Non-finite worker matrix product')
        return result

    def precondition(self, x):
        result = self.factor.solve(np.asarray(x,dtype=np.float64))
        if not np.isfinite(result).all():
            raise ArithmeticError('Non-finite preconditioner result')
        return result

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(); self.pool = None
        if self.accelerator is not None:
            self.accelerator.close(); self.accelerator = None


def compact_rows(matrix, lo, hi):
    part = matrix[lo:hi,:].tocsr()
    columns = np.unique(part.indices)
    compact = sparse.csr_matrix((part.data.copy(),np.searchsorted(columns,part.indices),part.indptr.copy()),
                               shape=(hi-lo,len(columns)))
    return compact, columns, matrix[lo:hi,lo:hi].tocsc()


def krylov(matrix, rhs, matvec, precondition, restart=30, maxiter=100, rtol=1e-10):
    if not np.isfinite(rhs).all():
        raise ValueError('Non-finite linear right hand side')
    n = len(rhs); count = [0]
    def callback(_): count[0] += 1
    op = LinearOperator((n,n),matvec=matvec,dtype=np.float64)
    pc = LinearOperator((n,n),matvec=precondition,dtype=np.float64)
    x, info = gmres(op,rhs,M=pc,rtol=rtol,atol=1e-12,restart=min(restart,n),maxiter=maxiter,
                   callback=callback,callback_type='pr_norm')
    # Independently recompute the TRUE residual on the coordinator CPU. This
    # guards device errors, response corruption and preconditioned-norm confusion.
    residual = np.linalg.norm(matrix @ x-rhs)
    bound = max(1e-12,rtol*np.linalg.norm(rhs))
    if info != 0 or not np.isfinite(x).all() or not np.isfinite(residual) or residual > bound*1.05:
        raise ArithmeticError(f'Linear solve failed: GMRES info={info}, true residual={residual}, bound={bound}')
    return x, {'krylov_iterations':count[0], 'linear_residual':float(residual)}


class LocalBackend:
    def __init__(self, threads=1, accelerator='cpu', restart=30, method='gmres'):
        if threads < 1 or threads > 256 or restart < 2 or method not in ('gmres','direct'):
            raise ValueError('Invalid thread count, restart or linear method')
        if accelerator not in ('cpu','opencl','opengl'):
            raise ValueError('Unknown accelerator')
        if accelerator == 'opengl' and threads != 1:
            raise ValueError('OpenGL contexts require --threads 1; use one worker process per GPU')
        if method == 'direct' and accelerator != 'cpu':
            raise ValueError('Direct sparse reference solve is CPU only')
        self.threads,self.accelerator,self.restart,self.method = threads,accelerator,restart,method
        self.stats = {}

    def solve(self, matrix, rhs):
        start = time.perf_counter()
        if self.method == 'direct':
            x = splu(matrix.tocsc()).solve(rhs)
            residual = np.linalg.norm(matrix @ x-rhs)
            if not np.isfinite(x).all() or residual > max(1e-12,1e-10*np.linalg.norm(rhs))*1.05:
                raise ArithmeticError('Sparse direct reference failed true-residual check')
            self.stats = {'krylov_iterations':0,'linear_residual':float(residual),'linear_seconds':time.perf_counter()-start}
            return x
        ranges = partitions(len(rhs),self.threads)
        with ExitStack() as resources:
            shards, columns = [], []
            for lo,hi in ranges:
                compact,col,diag = compact_rows(matrix,lo,hi)
                worker = RowShard(compact,diag,self.accelerator)
                resources.callback(worker.close); shards.append(worker); columns.append(col)
            pool = resources.enter_context(ThreadPoolExecutor(max_workers=len(ranges)))
            def product(x):
                return np.concatenate(list(pool.map(lambda k: shards[k].matvec(x[columns[k]]),range(len(shards)))))
            def precondition(x):
                return np.concatenate(list(pool.map(lambda k: shards[k].precondition(x[ranges[k][0]:ranges[k][1]]),range(len(shards)))))
            x,self.stats = krylov(matrix,rhs,product,precondition,self.restart)
        self.stats['linear_seconds'] = time.perf_counter()-start
        return x

    def close(self):
        pass
