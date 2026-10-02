"""Content-validated checkpoints partitioned by inputs, methods and versions."""
import json
from pathlib import Path
from .io_utils import digest, write_json


def valid(path, signature):
    path = Path(path)
    marker = path/'completion.json'
    if not marker.is_file(): return False
    metadata = json.loads(marker.read_text(encoding='utf-8'))
    if metadata.get('signature') != signature: return False
    return all((path/name).is_file() and digest(path/name) == sha
               for name, sha in metadata['files'].items())


def complete(path, signature):
    path = Path(path)
    write_json(dict(signature=signature, files={p.name:digest(p) for p in sorted(path.iterdir())
               if p.is_file() and p.name != 'completion.json'}), path/'completion.json')


def prepare(path, signature, force=False):
    path = Path(path)
    if not force and valid(path, signature): return True
    path.mkdir(parents=True, exist_ok=True)
    # These are exclusively generated files in a resolved checkpoint directory.
    for p in path.iterdir():
        if p.is_file(): p.unlink()
    return False
