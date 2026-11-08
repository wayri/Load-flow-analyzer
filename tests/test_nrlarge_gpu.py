"""Optional hardware checks. Absence is an explicit skip, never a CPU fallback."""
import unittest
import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse

from nrlarge.accelerators import (
    AcceleratorUnavailable, accelerator_info, create_accelerator, _matrix,
)


class Validation(unittest.TestCase):
    def test_rejects_bad_matrices(self):
        for matrix in (np.eye(2), sparse.csr_matrix([[complex(1, 2)]]),
                       sparse.csr_matrix([[np.inf]])):
            with self.assertRaises(ValueError):
                _matrix(matrix)
        with self.assertRaises(ValueError):
            create_accelerator("magic", sparse.eye(1))

    def test_normalizes_duplicates_and_int64(self):
        matrix = sparse.csr_matrix((np.array([2., 3.]), np.array([0, 0], dtype=np.int64),
                                    np.array([0, 2], dtype=np.int64)), shape=(1, 1))
        normalized = _matrix(matrix)
        self.assertEqual(normalized.nnz, 1)
        self.assertEqual(normalized[0, 0], 5.)


class GPUContract:
    kind = None

    @classmethod
    def setUpClass(cls):
        try:
            value = create_accelerator(cls.kind, sparse.eye(1, format="csr"))
        except AcceleratorUnavailable as error:
            raise unittest.SkipTest(str(error)) from error
        else:
            value.close()

    def test_rectangular_and_repeated_products(self):
        rng = np.random.default_rng(481)
        matrix = sparse.random(53, 79, density=.17, random_state=rng, format="csr")
        matrix.data -= .5
        matrix.indices = matrix.indices.astype(np.int64)
        matrix.indptr = matrix.indptr.astype(np.int64)
        backend = create_accelerator(self.kind, matrix)
        try:
            for _ in range(4):
                vector = rng.normal(size=matrix.shape[1])
                np.testing.assert_allclose(backend.matvec(vector), matrix @ vector,
                                           rtol=2e-13, atol=2e-13)
        finally:
            backend.close()

    def test_empty_rows_and_zero_dimensions(self):
        for shape in ((0, 5), (5, 0), (4, 9)):
            backend = create_accelerator(self.kind, sparse.csr_matrix(shape))
            try:
                np.testing.assert_array_equal(backend.matvec(np.zeros(shape[1])), np.zeros(shape[0]))
            finally:
                backend.close()

    def test_input_validation_and_lifecycle(self):
        backend = create_accelerator(self.kind, sparse.eye(2, format="csr"))
        try:
            for vector in ([1.], [[1., 2.]], [1., np.nan], [complex(1, 2), 1.]):
                with self.assertRaises(ValueError):
                    backend.matvec(vector)
        finally:
            backend.close()
            backend.close()
        with self.assertRaises(RuntimeError):
            backend.matvec([1., 2.])

    def test_nonfinite_output_is_failure(self):
        backend = create_accelerator(self.kind, sparse.csr_matrix([[1e308]]))
        try:
            with self.assertRaises(ArithmeticError):
                backend.matvec([1e308])
        finally:
            backend.close()


class OpenCL(GPUContract, unittest.TestCase):
    kind = "opencl"


class OpenGL(GPUContract, unittest.TestCase):
    kind = "opengl"

    def test_context_restoration_and_worker_thread(self):
        first = create_accelerator("opengl", sparse.eye(3, format="csr"))
        second = create_accelerator("opengl", 2 * sparse.eye(3, format="csr"))
        provider = first._glfw
        before = provider.get_current_context()
        try:
            provider.make_context_current(first._window)
            np.testing.assert_allclose(second.matvec([1., 2., 3.]), [2., 4., 6.])
            current = provider.get_current_context()
            if isinstance(first._window, dict):
                self.assertEqual(current, first._window)
            else:
                self.assertEqual(ctypes.cast(current, ctypes.c_void_p).value,
                                 ctypes.cast(first._window, ctypes.c_void_p).value)
            provider.make_context_current(None)
            with ThreadPoolExecutor(max_workers=1) as pool:
                np.testing.assert_allclose(pool.submit(first.matvec, [3., 4., 5.]).result(timeout=10),
                                           [3., 4., 5.])
                with self.assertRaises(AcceleratorUnavailable):
                    pool.submit(create_accelerator, "opengl", sparse.eye(1)).result(timeout=10)
        finally:
            provider.make_context_current(before)
            first.close()
            second.close()
        if hasattr(provider, "windows"):
            self.assertFalse(provider.windows, "Native WGL leaked a context/window")
            self.assertIsNone(provider.atom, "Native WGL leaked its owned window class")


if __name__ == "__main__":
    unittest.main(verbosity=2)
