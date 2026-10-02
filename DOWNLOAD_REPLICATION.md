# Download and verify the complete replication package

Release: https://github.com/Balint-Mariam/paperqfev2/releases/tag/replication-package-v1

1. Download `RELEASE_ASSETS_MANIFEST.json` and every archive file listed below
   into the same empty directory. Do not rename the files.
2. From that directory, run the Python command below. It verifies every part,
   joins split parts if necessary, verifies the complete archive, and extracts
   `replication/`. Extraction restores the raw options CSV, all six original
   inputs, generated data, execution caches, code, and audit evidence.
3. Follow `replication/README_REPLICATION.md` to install dependencies and run.

The complete archive needs approximately 10.3 GB after extraction. Allow space
for the downloaded archive and the assembled ZIP as well (at least 20 GB free). Stage 2 remains numerically uncertified.

## Archive files

- `QFE_replication_full.zip.part001` (1,900,000,000 bytes)
- `QFE_replication_full.zip.part002` (1,049,372,966 bytes)

## Verify and extract (Python 3.12)

Save this code as `extract_package.py` in the download directory and execute
`py -3.12 extract_package.py` (or `python extract_package.py`):

```python
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

manifest = json.loads(Path('RELEASE_ASSETS_MANIFEST.json').read_text())

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

parts = manifest['archive_parts']
for item in parts:
    p = Path(item['name'])
    if p.stat().st_size != item['size_bytes'] or sha256(p) != item['sha256']:
        raise RuntimeError(f'Corrupt download: {p}')
archive = Path(manifest['archive_name'])
if len(parts) > 1:
    if archive.exists():
        raise RuntimeError('Remove or move an existing assembled archive first')
    with archive.open('wb') as out:
        for item in parts:
            with Path(item['name']).open('rb') as source:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
if sha256(archive) != manifest['archive_sha256']:
    raise RuntimeError('Complete archive checksum mismatch')
with zipfile.ZipFile(archive) as z:
    base = Path.cwd().resolve()
    for name in z.namelist():
        if not (base / name).resolve().is_relative_to(base):
            raise RuntimeError('Unsafe archive path')
    if Path('replication').exists():
        raise RuntimeError('Extract into an empty directory to preserve existing files')
    z.extractall('.')
print('Verified and extracted replication/')
```

All checksum failures stop extraction. The full internal file inventory is
`replication/REPLICATION_DELIVERY_MANIFEST.json`; original input hashes are in
`replication/inputs/INPUTS_MANIFEST.json`.
