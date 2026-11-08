"""Explicit trusted workers: verified TLS, token authentication, bounded numeric RPC.

No pickle, arbitrary Python, shell commands, filesystem paths or remote installation
are accepted. Only sparse row configuration, matvec and preconditioner operations.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
from pathlib import Path
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import numpy as np
from scipy import sparse
from .linear import RowShard, compact_rows, partitions, krylov

PROTOCOL = 1
MAX_FRAME = 64*1024*1024
CHUNK_THRESHOLD = 8*1024*1024
CHUNK_ELEMENTS = 131072


def token_from_file(path):
    token = Path(path).read_text(encoding='ascii').strip()
    if len(token) < 32 or len(token) > 1024 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError('Worker token must be 32..1024 ASCII characters without whitespace')
    return token


def pack_array(array, dtype='<f8'):
    a = np.ascontiguousarray(array,dtype=dtype)
    if a.ndim != 1 or (dtype == '<f8' and not np.isfinite(a).all()):
        raise ValueError('RPC arrays must be finite and one-dimensional')
    return {'dtype':dtype,'count':len(a),'data':base64.b64encode(a.tobytes()).decode('ascii')}


def unpack_array(value, dtype='<f8', count=None):
    if not isinstance(value,dict) or set(value) != {'dtype','count','data'} or value['dtype'] != dtype:
        raise ValueError('Invalid RPC array schema or byte order')
    n = value['count']
    if type(n) is not int or n < 0 or n > MAX_FRAME//8 or (count is not None and n != count):
        raise ValueError('Invalid RPC array length')
    if not isinstance(value['data'],str) or len(value['data']) > (MAX_FRAME*4//3)+4:
        raise ValueError('RPC encoded array exceeds frame bound')
    data = base64.b64decode(value['data'],validate=True)
    if len(data) != n*8:
        raise ValueError('Truncated or oversized RPC array')
    result = np.frombuffer(data,dtype=dtype).copy()
    if dtype == '<f8' and not np.isfinite(result).all():
        raise ValueError('Non-finite RPC array')
    return result


def pack_matrix(matrix):
    m = matrix.tocsr()
    return {'shape':list(m.shape),'data':pack_array(m.data),
            'indices':pack_array(m.indices,'<i8'),'indptr':pack_array(m.indptr,'<i8')}


def csr_parts(shape, data, index, ptr):
    if len(ptr) != shape[0]+1 or len(index) != len(data) or ptr[0] != 0 or ptr[-1] != len(data) or np.any(ptr[1:] < ptr[:-1]) or np.any(index < 0) or np.any(index >= shape[1]) or not np.isfinite(data).all():
        raise ValueError('Invalid CSR structure')
    m = sparse.csr_matrix((data,index,ptr),shape=tuple(shape))
    m.sum_duplicates(); m.sort_indices()
    return m


def unpack_matrix(value):
    if not isinstance(value,dict) or set(value) != {'shape','data','indices','indptr'}:
        raise ValueError('Invalid sparse matrix schema')
    shape = value['shape']
    if not isinstance(shape,list) or len(shape) != 2 or any(type(v) is not int or v < 0 or v > MAX_FRAME//8 for v in shape):
        raise ValueError('Invalid sparse matrix shape')
    data = unpack_array(value['data']); index = unpack_array(value['indices'],'<i8',len(data))
    ptr = unpack_array(value['indptr'],'<i8',shape[0]+1)
    return csr_parts(shape,data,index,ptr)


class WorkerState:
    """One exclusive leased job per process; OpenGL stays on the server thread."""
    def __init__(self, accelerator='cpu', memory_gib=1, lease_seconds=300, threads=1):
        if memory_gib <= 0 or not np.isfinite(memory_gib) or lease_seconds <= 0 or not np.isfinite(lease_seconds):
            raise ValueError('Invalid worker memory/lease bound')
        if type(threads) is not int or not 1 <= threads <= 256 or accelerator not in ('cpu','opencl','opengl'):
            raise ValueError('Invalid worker threads or accelerator')
        if accelerator == 'opengl' and threads != 1:
            raise ValueError('OpenGL worker requires threads=1')
        self.accelerator,self.memory_gib,self.lease = accelerator,memory_gib,lease_seconds
        self.threads = threads
        self.worker = None; self.job = None; self.epoch = -1; self.seq = -1; self.touched = 0
        self.staging = None

    def close(self):
        if self.worker is not None:
            self.worker.close()
        self.worker = None; self.job = None; self.epoch = -1; self.seq = -1
        self.staging = None

    def handle(self, request):
        if not isinstance(request,dict) or request.get('protocol') != PROTOCOL:
            raise ValueError('Unsupported worker protocol')
        job,epoch,seq,op = (request.get(k) for k in ('job','epoch','seq','op'))
        if not isinstance(job,str) or len(job) != 32 or any(c not in '0123456789abcdef' for c in job) or type(epoch) is not int or epoch < 0 or type(seq) is not int or seq < 0:
            raise ValueError('Invalid job/epoch/sequence')
        if self.job is not None and time.monotonic()-self.touched > self.lease:
            self.close()
        if op in ('configure','configure_begin'):
            if self.job is not None and (self.job != job or epoch <= self.epoch):
                raise ValueError('Worker busy or stale Jacobian epoch')
            if seq != 0:
                raise ValueError('Configuration sequence must be zero')
            if op == 'configure_begin':
                shape,nnz,dnnz = request.get('shape'),request.get('nnz'),request.get('diagonal_nnz')
                if not isinstance(shape,list) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape) or type(nnz) is not int or type(dnnz) is not int or min(nnz,dnnz) < 0:
                    raise ValueError('Invalid staged matrix shape/counts')
                rows = shape[0]
                estimated = (16*nnz+8*(rows+1))*3+(16*dnnz+8*(rows+1))*12
                if estimated > self.memory_gib*2**30:
                    raise MemoryError('Staged matrix/factor estimate exceeds worker RAM budget')
                self.close()
                specs = {'m_data':(nnz,'<f8'),'m_indices':(nnz,'<i8'),'m_indptr':(rows+1,'<i8'),
                         'd_data':(dnnz,'<f8'),'d_indices':(dnnz,'<i8'),'d_indptr':(rows+1,'<i8')}
                self.staging = {'shape':shape,'arrays':{k:np.empty(n,dtype=d) for k,(n,d) in specs.items()},
                                'next':{k:0 for k in specs}}
                self.job,self.epoch,self.seq = job,epoch,0
                self.touched = time.monotonic()
                return {'protocol':PROTOCOL,'job':job,'epoch':epoch,'seq':seq,'value':{'staging':True}}
            matrix = unpack_matrix(request['matrix']); diagonal = unpack_matrix(request['diagonal'])
            if diagonal.shape != (matrix.shape[0],matrix.shape[0]) or matrix.shape[0] < 1:
                raise ValueError('Inconsistent worker diagonal block')
            estimated = (matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes)*3 + (diagonal.data.nbytes+diagonal.indices.nbytes+diagonal.indptr.nbytes)*12
            if estimated > self.memory_gib*2**30:
                raise MemoryError('Worker matrix/factor estimate exceeds configured RAM budget')
            # Keep old lease until replacement is validated; on failure client aborts.
            self.close()
            self.worker = RowShard(matrix,diagonal,self.accelerator,threads=self.threads)
            self.job,self.epoch,self.seq = job,epoch,0
            value = {'rows':matrix.shape[0],'columns':matrix.shape[1],'accelerator':self.accelerator}
        elif op in ('configure_chunk','configure_commit'):
            if self.job != job or self.epoch != epoch or seq != self.seq+1 or self.staging is None:
                raise ValueError('Missing staged job, stale epoch or out-of-order chunk')
            staged = self.staging
            if op == 'configure_chunk':
                name,offset = request.get('array'),request.get('offset')
                if name not in staged['arrays'] or type(offset) is not int or offset != staged['next'][name]:
                    raise ValueError('Duplicate, missing or out-of-order numeric chunk')
                dest = staged['arrays'][name]
                chunk = unpack_array(request['values'],dtype=dest.dtype.str)
                if len(chunk) < 1 or len(chunk) > CHUNK_ELEMENTS or offset+len(chunk) > len(dest):
                    raise ValueError('Invalid staged numeric chunk size')
                dest[offset:offset+len(chunk)] = chunk
                staged['next'][name] += len(chunk)
                value = {'received':len(chunk)}
            else:
                if any(staged['next'][k] != len(v) for k,v in staged['arrays'].items()):
                    raise ValueError('Cannot commit an incomplete sparse matrix')
                arrays = staged['arrays']; shape = staged['shape']
                matrix = csr_parts(shape,arrays['m_data'],arrays['m_indices'],arrays['m_indptr'])
                diagonal = csr_parts([shape[0],shape[0]],arrays['d_data'],arrays['d_indices'],arrays['d_indptr'])
                self.worker = RowShard(matrix,diagonal,self.accelerator,threads=self.threads)
                self.staging = None
                value = {'rows':shape[0],'columns':shape[1],'accelerator':self.accelerator}
            self.seq = seq
        else:
            valid_sequence = seq == self.seq+1 or (op == 'release' and seq > self.seq)
            if self.job != job or self.epoch != epoch or not valid_sequence or (self.worker is None and not (op == 'release' and self.staging is not None)):
                raise ValueError('Missing job, stale epoch or out-of-order request')
            if op == 'matvec':
                vector = unpack_array(request['vector'],count=self.worker.matrix.shape[1])
                value = pack_array(self.worker.matvec(vector))
            elif op == 'precondition':
                vector = unpack_array(request['vector'],count=self.worker.matrix.shape[0])
                value = pack_array(self.worker.precondition(vector))
            elif op == 'release':
                self.close(); value = {'released':True}
            else:
                raise ValueError('Unknown numeric worker operation')
            if op != 'release': self.seq = seq
        self.touched = time.monotonic()
        return {'protocol':PROTOCOL,'job':job,'epoch':epoch,'seq':seq,'value':value}


def make_server(bind, port, token, state, cert=None, key=None, loopback_http=False, socket_timeout=30):
    if len(token) < 32:
        raise ValueError('A strong worker token is required')
    if not np.isfinite(socket_timeout) or socket_timeout <= 0:
        raise ValueError('Worker socket timeout must be positive and finite')
    loopback = ipaddress.ip_address(bind).is_loopback
    if not cert or not key:
        if not (loopback and loopback_http):
            raise ValueError('Remote workers require a TLS certificate and private key; HTTP is loopback-test-only')
    context = None
    if cert and key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert,key)
    class NumericServer(HTTPServer):
        def get_request(self):
            connection,address = self.socket.accept()
            connection.settimeout(socket_timeout)
            if context is not None:
                try:
                    connection = context.wrap_socket(connection,server_side=True)
                except Exception:
                    connection.close()
                    raise
            return connection,address
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass  # never log authorization headers or numeric payloads
        def do_POST(self):
            try:
                if self.path != '/v1' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.send_error(403,'Worker authentication failed'); return
                if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Require bounded JSON Content-Length')
                length = int(self.headers.get('Content-Length','0'))
                if length <= 0 or length > min(MAX_FRAME,int(state.memory_gib*2**30/4)):
                    raise ValueError('Worker request exceeds frame limit')
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Truncated request')
                result = state.handle(json.loads(body))
                response = json.dumps(result,allow_nan=False,separators=(',',':')).encode()
                if len(response) > MAX_FRAME:
                    raise ValueError('Worker response exceeds frame limit')
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(response))); self.end_headers(); self.wfile.write(response)
            except Exception as error:
                self.send_error(400,str(error)[:300])
    server = NumericServer((bind,port),Handler)
    server.timeout = 1
    return server


class RemoteWorker:
    def __init__(self, url, token, cafile=None, timeout=60, allow_loopback_http=False):
        parsed = urllib.parse.urlparse(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/'):
            raise ValueError('Worker endpoint must be a bare HTTPS origin')
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and allow_loopback_http and parsed.hostname in ('127.0.0.1','localhost','::1')):
            raise ValueError('Require verified HTTPS; plain HTTP is explicit loopback-test-only')
        if not parsed.hostname or timeout <= 0 or not np.isfinite(timeout):
            raise ValueError('Invalid worker endpoint or timeout')
        self.url,self.token,self.timeout = url.rstrip('/')+'/v1',token,timeout
        context = ssl.create_default_context(cafile=cafile)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # Refuse redirects, proxies and TLS verification bypasses for numeric workers.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*_): return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=context))
        self.job = uuid.uuid4().hex; self.epoch = -1; self.seq = -1; self.active = False

    def request(self, op, **payload):
        self.seq += 1
        request = dict(protocol=PROTOCOL,job=self.job,epoch=self.epoch,seq=self.seq,op=op,**payload)
        body = json.dumps(request,allow_nan=False,separators=(',',':')).encode()
        if len(body) > MAX_FRAME:
            raise MemoryError('Worker frame exceeds 64 MiB; use more/smaller partitions')
        req = urllib.request.Request(self.url,data=body,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        try:
            with self.opener.open(req,timeout=self.timeout) as response:
                content = response.read(MAX_FRAME+1)
                if len(content) > MAX_FRAME: raise ValueError('Oversized worker response')
                result = json.loads(content)
        except urllib.error.HTTPError as error:
            error.close()
            raise RuntimeError(f'Worker numeric RPC failed ({op}); solve aborted') from error
        except Exception as error:
            raise RuntimeError(f'Worker numeric RPC failed ({op}); solve aborted') from error
        if any(result.get(k) != request[k] for k in ('protocol','job','epoch','seq')) or 'value' not in result:
            raise ValueError('Stale, mismatched or malformed worker response')
        return result['value']

    def configure(self, matrix, diagonal):
        self.epoch += 1; self.seq = -1
        estimated = (matrix.nnz+diagonal.nnz)*24+(matrix.shape[0]+diagonal.shape[0]+2)*12
        if estimated < CHUNK_THRESHOLD:
            result = self.request('configure',matrix=pack_matrix(matrix),diagonal=pack_matrix(diagonal))
        else:
            diagonal = diagonal.tocsr()
            self.request('configure_begin',shape=list(matrix.shape),nnz=matrix.nnz,diagonal_nnz=diagonal.nnz)
            self.active = True
            for prefix,m in (('m',matrix),('d',diagonal)):
                for name,array,dtype in (('data',m.data,'<f8'),('indices',m.indices,'<i8'),('indptr',m.indptr,'<i8')):
                    for offset in range(0,len(array),CHUNK_ELEMENTS):
                        chunk = array[offset:offset+CHUNK_ELEMENTS]
                        received = self.request('configure_chunk',array=prefix+'_'+name,offset=offset,values=pack_array(chunk,dtype))
                        if received.get('received') != len(chunk):
                            raise ValueError('Worker did not acknowledge complete numeric chunk')
            result = self.request('configure_commit')
        self.active = True
        if result.get('rows') != matrix.shape[0] or result.get('columns') != matrix.shape[1]:
            raise ValueError('Worker acknowledged wrong row/ghost-column coverage')

    def close(self):
        if self.active:
            try: self.request('release')
            finally: self.active = False


class ClusterBackend:
    def __init__(self, endpoints, token, cafile=None, restart=30, timeout=60, allow_loopback_http=False):
        if not 1 <= len(endpoints) <= 20 or len(set(endpoints)) != len(endpoints):
            raise ValueError('Configure 1..20 unique worker origins')
        self.workers = [RemoteWorker(e,token,cafile,timeout,allow_loopback_http) for e in endpoints]
        self.restart,self.stats = restart,{}

    def solve(self, matrix, rhs):
        start = time.perf_counter()
        ranges = partitions(len(rhs),len(self.workers))
        workers = self.workers[:len(ranges)]
        columns = []
        # Each worker gets only its owned CSR rows and compact neighbor columns.
        # Acknowledged intervals are fixed locally; no partial/overlapping coverage.
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            uploads = []
            for worker,(lo,hi) in zip(workers,ranges):
                compact,col,diag = compact_rows(matrix,lo,hi); columns.append(col)
                uploads.append(pool.submit(worker.configure,compact,diag))
            for upload in uploads: upload.result()
            def product(x):
                def call(k):
                    return unpack_array(workers[k].request('matvec',vector=pack_array(x[columns[k]])),count=ranges[k][1]-ranges[k][0])
                return np.concatenate(list(pool.map(call,range(len(workers)))))
            def precondition(x):
                def call(k):
                    lo,hi = ranges[k]
                    return unpack_array(workers[k].request('precondition',vector=pack_array(x[lo:hi])),count=hi-lo)
                return np.concatenate(list(pool.map(call,range(len(workers)))))
            result,self.stats = krylov(matrix,rhs,product,precondition,self.restart)
        self.stats.update(linear_seconds=time.perf_counter()-start,workers=len(workers))
        return result

    def close(self):
        failures = []
        for worker in self.workers:
            try: worker.close()
            except Exception as error: failures.append(str(error))
        if failures:
            raise RuntimeError('Worker release failed; leases expire automatically: '+ '; '.join(failures))
