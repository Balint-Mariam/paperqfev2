"""Runtime identity, pinned dependencies, and mirrored terminal logging."""
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from .config import PACKAGE
from .io_utils import digest, token, write_json


def identity():
    requirements = {}
    for line in (PACKAGE / 'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            requirements[name] = version
    actual = {name: importlib.metadata.version(name) for name in requirements}
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), versions=actual, pinned=requirements,
                versions_match=actual == requirements,
                OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'))


def code_identity():
    return token({str(p.relative_to(PACKAGE)): digest(p) for p in sorted(PACKAGE.rglob('*.py'))
                  if 'outputs' not in p.parts and '__pycache__' not in p.parts})


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def start_log(config, stage):
    path = config['output_root'] / 'logs' / f'stage{stage}.log'
    log = path.open('w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    info = identity()
    write_json(info, config['output_root'] / 'diagnostics' / f'stage{stage}_environment.json')
    if not info['versions_match']:
        raise RuntimeError('Environment differs from requirements.txt; install the pinned environment before reproduction')
    return info


def end_log():
    """Detach both mirrored streams before closing their shared file once."""
    tees=[(name,getattr(sys,name)) for name in ['stdout','stderr']]
    logs={}
    for name,tee in tees:
        if hasattr(tee,'log'):
            if not tee.log.closed:tee.flush()
            logs[id(tee.log)]=tee.log
            setattr(sys,name,tee.terminal)
    for log in logs.values():
        if not log.closed:log.close()
