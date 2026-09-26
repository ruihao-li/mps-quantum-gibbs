"""Portable release I/O and process setup; no original-tree dependencies."""
from __future__ import annotations

# Must precede every numerical import, including in spawned processes.
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np


def jsonable(value):
    """Convert nested NumPy values and paths to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def encoded(value):
    """Encode sorted, indented JSON with a final newline and no nonfinite numbers."""
    return (json.dumps(jsonable(value), sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def digest(value):
    """Return the SHA-256 digest of the canonical JSON encoding."""
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    """Return a file's SHA-256 digest using bounded-size binary reads."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parameter_hash(theta):
    """Hash parameter bytes after conversion to a float64 NumPy array."""
    return hashlib.sha256(np.asarray(theta, dtype=np.float64).tobytes()).hexdigest()


def read(path):
    """Read a JSON value without independently verifying its integrity."""
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    """Publish JSON atomically without replacing a different existing record.

    An identical existing encoding is a no-op. Otherwise, flush a temporary
    file and hard-link it into place; differing existing bytes raise
    RuntimeError, and a concurrent target creation fails without clobbering.
    """
    path = Path(path)
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing changed existing record: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def environment():
    """Record dependency versions, numerical backend, and controlled process settings.

    Include contraction, autodiff, JIT, and circuit dependencies so that
    environment changes invalidate campaign validation and resume bindings.
    """
    versions = {}
    # Include contraction planners, JIT, autodiff, circuit and BLAS-adjacent
    # dependencies, not only the headline libraries. Their changes invalidate
    # resume/validation even when NumPy/JAX versions themselves are unchanged.
    for package in ("numpy", "scipy", "jax", "jaxlib", "optax", "chex",
                    "ml-dtypes", "opt-einsum", "qiskit", "quimb", "qiskit-quimb",
                    "cotengra", "autoray", "numba", "llvmlite", "networkx",
                    "psutil", "cytoolz", "toolz", "absl-py",
                    "rustworkx", "dill", "stevedore", "tqdm"):
        versions[package] = importlib.metadata.version(package)
    import jax
    import scipy
    controlled = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "JAX_ENABLE_X64",
                  "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS", "JAX_PLATFORM_NAME",
                  "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION")
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=versions, threads_per_process=1,
                jax_enable_x64=bool(jax.config.read("jax_enable_x64")),
                jax_backend=jax.default_backend(),
                jax_devices=[dict(platform=d.platform, kind=d.device_kind) for d in jax.devices()],
                numerical_environment={name: os.environ.get(name) for name in controlled},
                numpy_build_configuration=jsonable(getattr(np.__config__, "CONFIG", {})),
                scipy_build_configuration=jsonable(getattr(scipy.__config__, "CONFIG", {})))


def sources():
    """Hash Python modules directly inside the release's src and experiments directories."""
    paths = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def output_path(raw, experiment):
    """Resolve an external output root and append the experiment subdirectory.

    Raise ValueError if the supplied root is the release directory or lies
    inside it, keeping numerical output outside the code-only bundle.
    """
    result = Path(raw).expanduser().resolve()
    if result == ROOT or ROOT in result.parents:
        raise ValueError("Numerical outputs must be outside the code-only release directory")
    return result / experiment


def verify_directory(path, binding):
    """Verify a cached directory's binding, exact inventory, and file hashes.

    Return the completion marker only after every listed basename matches
    its digest and no extra or missing entries remain. This checks integrity
    against the supplied binding, not the scientific validity of the arrays.
    """
    path = Path(path)
    marker = read(path / "complete.json")
    if marker["binding"] != binding:
        raise RuntimeError(f"Input/source/options binding differs: {path}")
    expected = set(marker["files"]) | {"complete.json"}
    if {p.name for p in path.iterdir()} != expected:
        raise RuntimeError(f"Unexpected or missing files: {path}")
    for name, expected_hash in marker["files"].items():
        if Path(name).name != name or sha(path / name) != expected_hash:
            raise RuntimeError(f"Corrupt completed output: {path / name}")
    return marker


def publish_directory(target, binding, record, arrays, *, record_name="record.json"):
    """Publish metadata and compressed arrays as a complete result directory.

    Args:
        target: Final result directory; an existing one is verified and reused.
        binding: Expected input/source/options digest for the completion marker.
        record: JSON-compatible metadata to publish.
        arrays: Mapping from NPZ basenames to named-array mappings.
        record_name: Metadata filename inside the result directory.

    Notes:
        Stage files and their checksum inventory in a hidden sibling, then
        rename it into place. Failed writes may leave that scratch directory.
        Concurrent launches into a shared output root are unsupported.
    """
    target = Path(target)
    if target.exists():
        verify_directory(target, binding)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    # A failed write leaves an explicitly hidden, uncommitted scratch directory.
    immutable_json(temporary / record_name, record)
    for name, values in arrays.items():
        if Path(name).name != name:
            raise ValueError("Array filename must be a basename")
        np.savez_compressed(temporary / name, **values)
    files = {p.name: sha(p) for p in temporary.iterdir()}
    immutable_json(temporary / "complete.json", dict(schema_version=1, binding=binding, files=files))
    # A nonempty existing directory cannot be replaced by rename. Concurrent
    # launch into one output root is unsupported and must not be attempted.
    temporary.rename(target)
