"""Content identities and deterministic machine-readable exports."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def token(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + '\n', encoding='utf-8')


def save(frame, path, sort=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort:
        frame = frame.sort_values(sort, kind='stable').reset_index(drop=True)
    frame.to_parquet(path, index=False, compression='zstd')
