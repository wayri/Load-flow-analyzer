"""Real spawned worker processes, bounded numeric RPC, and TLS trust contracts.

These tests launch only explicitly owned loopback listeners. No optional GPU,
remote machine, package installation, or unverified certificate is required.
"""
import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import multiprocessing
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nrlarge.cluster import (
    ClusterBackend, MAX_FRAME, PROTOCOL, RemoteWorker, WorkerState,
    make_server, pack_array, pack_matrix, token_from_file, unpack_array, unpack_matrix,
)
from nrlarge.linear import compact_rows
import nrlarge.cluster as cluster_module

TOKEN = "nrlarge-test-token-not-for-production-0123456789"


def task_temporary(prefix):
    scratch = Path(__file__).resolve().parents[2]/".task-check"
    scratch.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch)


def _serve_process(sender, stop, cert=None, key=None, socket_timeout=30):
    state = WorkerState(memory_gib=.1)
    server = None
    try:
        server = make_server("127.0.0.1", 0, TOKEN, state, cert, key,
                             loopback_http=cert is None, socket_timeout=socket_timeout)
        server.timeout = .1
        sender.send(("ready", server.server_address[1]))
        sender.close()
        while not stop.is_set():
            server.handle_request()
    except BaseException as error:
        try: sender.send(("error", repr(error)))
        except (BrokenPipeError, EOFError, OSError): pass
        raise
    finally:
        state.close()
        if server is not None: server.server_close()


@contextmanager
def processes(count=2, cert=None, key=None, socket_timeout=30):
    """Explicit spawn also exercises Windows' process-import behavior."""
    ctx = multiprocessing.get_context("spawn")
    owned, endpoints = [], []
    try:
        for _ in range(count):
            receiver, sender = ctx.Pipe(duplex=False)
            stop = ctx.Event()
            process = ctx.Process(target=_serve_process,
                                  args=(sender, stop, cert, key, socket_timeout))
            process.start(); sender.close()
            owned.append((process, stop))
            if not receiver.poll(15):
                raise RuntimeError("Owned loopback test worker did not start")
            status, value = receiver.recv(); receiver.close()
            if status != "ready": raise RuntimeError(value)
            endpoints.append(f"{'https' if cert else 'http'}://127.0.0.1:{value}")
        yield endpoints
    finally:
        for process, stop in owned: stop.set()
        for process, _ in owned:
            process.join(3)
            if process.is_alive():
                # The test owns this exact child; failed TLS/header clients may
                # otherwise prevent a single-threaded server from accepting again.
                process.terminate(); process.join(3)
            process.close()


def matrix_and_rhs(n=14):
    # Nonsymmetric, off-block couplings and odd numbers of unknown-pairs per shard.
    diagonal = np.linspace(6, 9, n)
    matrix = sparse.diags([-.8*np.ones(n-1), diagonal, .3*np.ones(n-1)],
                          [-1, 0, 1], format="lil")
    matrix[0, n-1] = .2; matrix[n-2, 1] = -.4
    matrix = matrix.tocsr()
    expected = np.linspace(-.7, 1.3, n)
    return matrix, matrix @ expected, expected


def configure_request(job="a"*32, epoch=0):
    matrix, _, _ = matrix_and_rhs(4)
    compact, _, diagonal = compact_rows(matrix, 0, 2)
    return dict(protocol=PROTOCOL, job=job, epoch=epoch, seq=0, op="configure",
                matrix=pack_matrix(compact), diagonal=pack_matrix(diagonal))


@contextmanager
def reply_server(reply):
    """Task-owned malformed/redirecting HTTP peer; never contacts the Internet."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            result, status, extra = reply(request)
            data = json.dumps(result, allow_nan=True).encode()
            self.send_response(status)
            for name, value in extra.items(): self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            # A redirect-following client would reach here; record the attempt
            # through the same callback so refusal can be verified, not inferred.
            result, status, extra = reply({"unexpected_redirect_get":True})
            data = json.dumps(result).encode()
            self.send_response(status)
            for name, value in extra.items(): self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try: yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


class ClusterProcesses(unittest.TestCase):
    def test_two_real_workers_use_multiframe_staged_upload(self):
        matrix, rhs, expected = matrix_and_rhs()
        original_request = RemoteWorker.request
        observed_operations = []
        def observe(worker, op, **payload):
            observed_operations.append(op)
            return original_request(worker, op, **payload)
        with processes(2) as endpoints:
            backend = ClusterBackend(endpoints, TOKEN, restart=14, timeout=5,
                                     allow_loopback_http=True)
            try:
                # Tiny chunks exercise multiple offsets per array without
                # allocating multi-megabyte fixtures. Server bounds stay real.
                with mock.patch.object(cluster_module, "CHUNK_THRESHOLD", 0), \
                     mock.patch.object(cluster_module, "CHUNK_ELEMENTS", 2), \
                     mock.patch.object(RemoteWorker, "request", observe):
                    answer = backend.solve(matrix, rhs)
                assert_allclose(answer, spsolve(matrix, rhs), rtol=1e-10, atol=1e-11)
                assert_allclose(answer, expected, rtol=1e-10, atol=1e-11)
                self.assertEqual(observed_operations.count("configure_begin"), 2)
                self.assertEqual(observed_operations.count("configure_commit"), 2)
                self.assertGreater(observed_operations.count("configure_chunk"), 12)
                self.assertNotIn("configure", observed_operations)
            finally: backend.close()

    def test_two_spawned_workers_solve_and_new_jacobian_epoch(self):
        matrix, rhs, expected = matrix_and_rhs()
        with processes(2) as endpoints:
            backend = ClusterBackend(endpoints, TOKEN, restart=14, timeout=5,
                                     allow_loopback_http=True)
            try:
                answer = backend.solve(matrix, rhs)
                assert_allclose(answer, spsolve(matrix, rhs), rtol=1e-10, atol=1e-11)
                assert_allclose(answer, expected, rtol=1e-10, atol=1e-11)
                self.assertEqual(backend.stats["workers"], 2)
                self.assertLessEqual(backend.stats["linear_residual"], 1e-9)
                changed = matrix + sparse.eye(matrix.shape[0]) * .25
                again = backend.solve(changed, changed @ expected)
                assert_allclose(again, expected, rtol=1e-10, atol=1e-11)
                self.assertTrue(all(worker.epoch == 1 for worker in backend.workers))
            finally: backend.close()

    def test_more_workers_than_unknown_pairs_and_released_lease(self):
        matrix, rhs, expected = matrix_and_rhs(2)
        with processes(2) as endpoints:
            backend = ClusterBackend(endpoints, TOKEN, allow_loopback_http=True)
            try:
                assert_allclose(backend.solve(matrix, rhs), expected, atol=1e-11)
                self.assertEqual(backend.stats["workers"], 1)
                self.assertFalse(backend.workers[1].active)
            finally: backend.close()
            # A distinct coordinator acquires the same workers after release.
            replacement = ClusterBackend(endpoints, TOKEN, allow_loopback_http=True)
            try: assert_allclose(replacement.solve(matrix, rhs), expected, atol=1e-11)
            finally: replacement.close()

    def test_authentication_failure_is_not_a_numeric_result(self):
        with processes(1) as endpoints:
            worker = RemoteWorker(endpoints[0], "incorrect-token-" + "x"*32,
                                  timeout=2, allow_loopback_http=True)
            matrix, _, _ = matrix_and_rhs(2)
            with self.assertRaises(RuntimeError): worker.configure(matrix, matrix)
            self.assertFalse(worker.active)

    def test_unreachable_worker_aborts_instead_of_partial_rows(self):
        with processes(1) as endpoints:
            unused = socket.socket(); unused.bind(("127.0.0.1", 0))
            unavailable = f"http://127.0.0.1:{unused.getsockname()[1]}"
            unused.close()
            backend = ClusterBackend([endpoints[0], unavailable], TOKEN,
                                     timeout=.3, allow_loopback_http=True)
            matrix, rhs, _ = matrix_and_rhs()
            with self.assertRaises(RuntimeError): backend.solve(matrix, rhs)
            backend.close()


class NumericProtocol(unittest.TestCase):
    def test_chunk_coverage_duplicate_gap_stale_and_incomplete_commit(self):
        matrix, _, _ = matrix_and_rhs(4)
        compact, _, diagonal = compact_rows(matrix, 0, 2)
        diagonal = diagonal.tocsr()
        state = WorkerState(memory_gib=.1)
        begin = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=0,
                     op="configure_begin", shape=list(compact.shape),
                     nnz=compact.nnz, diagonal_nnz=diagonal.nnz)
        arrays = {}
        for prefix, part in (("m", compact), ("d", diagonal)):
            arrays[prefix+"_data"] = (part.data, "<f8")
            arrays[prefix+"_indices"] = (part.indices, "<i8")
            arrays[prefix+"_indptr"] = (part.indptr, "<i8")
        try:
            state.handle(begin)
            first = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=1,
                         op="configure_chunk", array="m_data", offset=0,
                         values=pack_array(compact.data[:2]))
            self.assertEqual(state.handle(first)["value"]["received"], 2)
            for invalid in (dict(first, seq=2),
                            dict(first, seq=2, offset=3),
                            dict(first, seq=2, offset=2, epoch=1),
                            dict(first, offset=2),
                            dict(first, seq=2, offset=2, job="b"*32),
                            dict(first, seq=2, array="not_an_array")):
                with self.assertRaises(ValueError): state.handle(invalid)
            commit = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=2,
                          op="configure_commit")
            with self.assertRaises(ValueError): state.handle(commit)
            # Rejected chunks do not advance sequence or coverage. Complete the
            # remaining exact coverage and verify usable assembled row values.
            seq = 1
            for name, (array, dtype) in arrays.items():
                start = 2 if name == "m_data" else 0
                for offset in range(start, len(array), 2):
                    seq += 1
                    chunk = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=seq,
                                 op="configure_chunk", array=name, offset=offset,
                                 values=pack_array(array[offset:offset+2], dtype))
                    state.handle(chunk)
            state.handle(dict(commit, seq=seq+1))
            self.assertIsNone(state.staging)
            response = state.handle(dict(protocol=PROTOCOL, job="a"*32, epoch=0,
                seq=seq+2, op="matvec", vector=pack_array(np.ones(compact.shape[1]))))
            assert_allclose(unpack_array(response["value"]), compact @ np.ones(compact.shape[1]))
        finally: state.close()

    def test_staged_upload_budget_rejected_before_numeric_allocation(self):
        state = WorkerState(memory_gib=.001)
        begin = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=0,
                     op="configure_begin", shape=[1_000_000, 1_000_000],
                     nnz=3_000_000, diagonal_nnz=1_000_000)
        # Patch only allocation: a bad preflight must not allocate even one
        # numeric array, let alone the declared multi-megabyte transfer.
        with mock.patch.object(cluster_module.np, "empty", side_effect=AssertionError("Allocation before memory check")) as allocate:
            with self.assertRaises(MemoryError): state.handle(begin)
            allocate.assert_not_called()
        self.assertIsNone(state.staging)
        self.assertIsNone(state.worker)
        self.assertIsNone(state.job)

    def test_explicit_release_discards_an_incomplete_staged_job(self):
        state = WorkerState(memory_gib=.1)
        begin = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=0,
                     op="configure_begin", shape=[2, 2], nnz=2, diagonal_nnz=2)
        try:
            state.handle(begin)
            self.assertIsNotNone(state.staging)
            released = state.handle(dict(protocol=PROTOCOL, job="a"*32, epoch=0,
                                         seq=1, op="release"))
            self.assertTrue(released["value"]["released"])
            self.assertIsNone(state.staging)
            self.assertIsNone(state.worker)
            self.assertIsNone(state.job)
        finally: state.close()

    def test_exclusive_job_stale_epoch_sequence_and_lease(self):
        state = WorkerState(memory_gib=.1, lease_seconds=30)
        request = configure_request()
        try:
            state.handle(request)
            for modified in (dict(request), dict(request, job="b"*32),
                             dict(request, epoch=0, seq=1)):
                with self.assertRaises(ValueError): state.handle(modified)
            vector = pack_array(np.ones(4))
            valid = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=1,
                         op="matvec", vector=vector)
            response = state.handle(valid)
            self.assertEqual(response["seq"], 1)
            with self.assertRaises(ValueError): state.handle(valid)
            with self.assertRaises(ValueError): state.handle(dict(valid, epoch=1, seq=2))
            with self.assertRaises(ValueError): state.handle(dict(valid, seq=3))
            # Age the lease deterministically; avoid a timing-sensitive tiny
            # lease that could expire midway through checks on a busy CI host.
            state.touched -= 31
            state.handle(configure_request("b"*32))
            self.assertEqual(state.job, "b"*32)
        finally: state.close()

    def test_invalid_array_and_csr_descriptors(self):
        ordinary = pack_array(np.arange(3.))
        for descriptor in (dict(ordinary, dtype="<f4"), dict(ordinary, count=2),
                           dict(ordinary, count=True), dict(ordinary, count=-1),
                           dict(ordinary, data="not valid base64!"),
                           dict(ordinary, count=MAX_FRAME),
                           dict(ordinary, extra=1)):
            with self.assertRaises((ValueError, TypeError)): unpack_array(descriptor)
        nonfinite = dict(dtype="<f8", count=1,
                         data=base64.b64encode(np.array([np.nan], dtype="<f8").tobytes()).decode())
        with self.assertRaises(ValueError): unpack_array(nonfinite)
        matrix, _, _ = matrix_and_rhs(4)
        valid = pack_matrix(matrix)
        invalid = [dict(valid, shape=[4, 3]), dict(valid, shape=[-1, 4]),
                   dict(valid, indices=pack_array([-1]*matrix.nnz, "<i8")),
                   dict(valid, indptr=pack_array([0, 2, 1, 5, matrix.nnz], "<i8"))]
        for descriptor in invalid:
            with self.assertRaises(ValueError): unpack_matrix(descriptor)

    def test_dimensions_and_worker_memory_limit(self):
        state = WorkerState(memory_gib=.1)
        request = configure_request()
        request["diagonal"] = pack_matrix(sparse.eye(3))
        with self.assertRaises(ValueError): state.handle(request)
        request = configure_request(); state.handle(request)
        try:
            wrong = dict(protocol=PROTOCOL, job="a"*32, epoch=0, seq=1,
                         op="matvec", vector=pack_array([1., 2.]))
            with self.assertRaises(ValueError): state.handle(wrong)
        finally: state.close()
        tiny = WorkerState(memory_gib=1e-9)
        with self.assertRaises(MemoryError): tiny.handle(configure_request())
        self.assertIsNone(tiny.worker)

    def test_endpoints_must_be_explicit_trusted_origins(self):
        for endpoint in ("http://example.com", "http://127.0.0.2", "ftp://localhost",
                         "https://user:pass@localhost", "https://localhost/v1",
                         "https://localhost/?q=x", "https://localhost/#x"):
            with self.assertRaises(ValueError):
                RemoteWorker(endpoint, TOKEN, allow_loopback_http=True)
        with self.assertRaises(ValueError): RemoteWorker("http://127.0.0.1", TOKEN)
        for endpoints in ([], ["https://localhost"]*2,
                          [f"https://host-{i}.example" for i in range(21)]):
            with self.assertRaises(ValueError): ClusterBackend(endpoints, TOKEN)
        with self.assertRaises(ValueError):
            make_server("0.0.0.0", 0, TOKEN, WorkerState(), loopback_http=True)

    def test_stale_response_and_redirect_are_rejected(self):
        def stale(request):
            return dict(request, seq=request["seq"]+1, value={}), 200, {}
        with reply_server(stale) as endpoint:
            worker = RemoteWorker(endpoint, TOKEN, allow_loopback_http=True)
            worker.epoch = 0
            with self.assertRaises(ValueError): worker.request("matvec", vector=pack_array([1.]))
        redirected_hits = []
        def destination(request):
            redirected_hits.append(request)
            return {}, 200, {}
        with reply_server(destination) as target:
            with reply_server(lambda _: ({}, 302, {"Location":target+"/v1"})) as endpoint:
                worker = RemoteWorker(endpoint, TOKEN, allow_loopback_http=True)
                worker.epoch = 0
                with self.assertRaises(RuntimeError): worker.request("matvec", vector=pack_array([1.]))
            self.assertEqual(redirected_hits, [], "Numeric worker followed a redirect")

    def test_corrupt_numeric_worker_results_cannot_complete_a_solve(self):
        matrix, rhs, _ = matrix_and_rhs(4)
        for corruption in ("length", "nonfinite", "wrong_product"):
            with self.subTest(corruption=corruption):
                state = WorkerState(memory_gib=.1)
                def reply(request):
                    response = state.handle(request)
                    if request["op"] == "matvec":
                        if corruption == "length": response["value"] = pack_array(np.zeros(5))
                        elif corruption == "nonfinite":
                            response["value"] = {"dtype":"<f8", "count":4,
                                "data":base64.b64encode(np.full(4, np.nan, dtype="<f8").tobytes()).decode()}
                        else:
                            response["value"] = pack_array(-unpack_array(response["value"]))
                    return response, 200, {}
                with reply_server(reply) as endpoint:
                    backend = ClusterBackend([endpoint], TOKEN, allow_loopback_http=True)
                    try:
                        with self.assertRaises((ValueError, ArithmeticError)):
                            backend.solve(matrix, rhs)
                    finally: backend.close()
                state.close()

    def test_token_file_does_not_accept_short_or_whitespace_credentials(self):
        with task_temporary("nrlarge-token-test-") as directory:
            path = Path(directory)/"token.txt"
            for value in ("short", "x"*32+" y", "x"*1025):
                path.write_text(value, encoding="ascii")
                with self.assertRaises(ValueError): token_from_file(path)
            path.write_text(TOKEN+"\n", encoding="ascii")
            self.assertEqual(token_from_file(path), TOKEN)


def _certificate_pair(directory):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NRlarge loopback test")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now-timedelta(minutes=5)).not_valid_after(now+timedelta(days=1))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                  x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
                   .sign(key, hashes.SHA256()))
    certpath, keypath = Path(directory)/"cert.pem", Path(directory)/"key.pem"
    certpath.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    keypath.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                       serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return str(certpath), str(keypath)


class VerifiedTLS(unittest.TestCase):
    def test_real_tls_worker_trust_hostname_and_wrong_token(self):
        try: import cryptography  # noqa: F401
        except ImportError: self.skipTest("Optional cryptography unavailable for temporary test certificates")
        with task_temporary("nrlarge-tls-test-") as directory:
            cert, key = _certificate_pair(directory)
            with processes(1, cert, key) as endpoints:
                matrix, rhs, expected = matrix_and_rhs(4)
                backend = ClusterBackend(endpoints, TOKEN, cafile=cert, timeout=3)
                try:
                    with mock.patch.object(cluster_module, "CHUNK_THRESHOLD", 0), \
                         mock.patch.object(cluster_module, "CHUNK_ELEMENTS", 2):
                        assert_allclose(backend.solve(matrix, rhs), expected, atol=1e-11)
                finally: backend.close()
                untrusted = RemoteWorker(endpoints[0], TOKEN, timeout=2)
                with self.assertRaises(RuntimeError): untrusted.configure(matrix, matrix)
                wrong = RemoteWorker(endpoints[0], "x"*40, cafile=cert, timeout=2)
                with self.assertRaises(RuntimeError): wrong.configure(matrix, matrix)
                # A failed handshake/authentication must not disable the worker.
                retry = ClusterBackend(endpoints, TOKEN, cafile=cert, timeout=3)
                try: assert_allclose(retry.solve(matrix, rhs), expected, atol=1e-11)
                finally: retry.close()

    def test_idle_handshake_and_partial_headers_have_bounded_wait(self):
        try: import cryptography  # noqa: F401
        except ImportError: self.skipTest("Optional cryptography unavailable for temporary test certificates")
        matrix, rhs, expected = matrix_and_rhs(4)
        for tls in (False, True):
            with self.subTest(tls=tls), task_temporary("nrlarge-timeout-test-") as directory:
                cert, key = _certificate_pair(directory) if tls else (None, None)
                with processes(1, cert, key, socket_timeout=.2) as endpoints:
                    port = int(endpoints[0].rsplit(":", 1)[1])
                    with socket.create_connection(("127.0.0.1", port), timeout=1) as stalled:
                        if not tls: stalled.sendall(b"POST /v1 HTTP/1.1\r\nHost: localhost\r\n")
                        # Keep the offending socket open while another client
                        # waits. The server must bound the first client's wait.
                        backend = ClusterBackend(endpoints, TOKEN, cafile=cert,
                                                 timeout=3, allow_loopback_http=not tls)
                        try: assert_allclose(backend.solve(matrix, rhs), expected, atol=1e-11)
                        finally: backend.close()


if __name__ == "__main__":
    unittest.main()
