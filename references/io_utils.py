"""Immutable output and digest helpers; no repository-root discovery."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    """Return the hexadecimal SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path, payload):
    """Publish finite JSON and its digest without replacing different content.

    Each file is created atomically; the JSON and digest are not one atomic
    transaction. Identical existing bytes are accepted for resumable runs.
    """
    path = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    immutable_bytes(path, data)
    immutable_bytes(Path(str(path) + '.sha256'), (sha256(path) + '\n').encode())


def immutable_bytes(path, data):
    """Create a file atomically, accepting identical concurrent publication.

    Raises:
        FileExistsError: Existing or concurrently published bytes differ.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Refusing different existing output: {path}; choose a fresh data root')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(f'Concurrent different output: {path}')
    finally:
        temporary.unlink(missing_ok=True)


def read_verified(path):
    """Verify a JSON file's SHA-256 sidecar and reject nonfinite constants.

    Returns:
        The decoded JSON payload.

    Raises:
        ValueError: The digest differs or JSON contains NaN or infinity.
    """
    path = Path(path)
    if Path(str(path) + '.sha256').read_text().strip() != sha256(path):
        raise ValueError(f'Checksum mismatch: {path}')
    def reject_constant(value):
        """Reject a nonfinite JSON literal encountered by the decoder."""
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(path.read_text(), parse_constant=reject_constant)
