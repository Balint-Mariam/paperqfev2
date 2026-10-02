"""Portable configuration; locked methods and input/output paths are separate."""
import argparse
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def load_args(stage):
    parser = argparse.ArgumentParser(description=f'Frozen QFE replication stage {stage}')
    parser.add_argument('--config', type=Path, default=PACKAGE / 'config/replication_config.json')
    parser.add_argument('--raw-options-file', type=Path)
    parser.add_argument('--rates-file', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    locked=json.loads((PACKAGE/'manifests/locked_specification.json').read_text(encoding='utf-8'))
    effective={k:v for k,v in config.items() if k not in ['raw_options_file','rates_file','reference_root','output_root']}
    if effective!=locked:
        raise ValueError('Configuration changes the locked empirical specification; only input/output paths and worker count may change')
    if args.workers<1:raise ValueError('--workers must be positive')
    for key in ['raw_options_file', 'rates_file', 'output_root', 'reference_root']:
        override = getattr(args, key)
        path = override if override is not None else PACKAGE / config[key]
        config[key] = Path(path).resolve()
    config['workers'] = args.workers
    config['force'] = args.force
    config['config_path'] = args.config.resolve()
    for name in ['data', 'tables', 'figures', 'diagnostics', 'logs', 'excel', 'cache']:
        (config['output_root'] / name).mkdir(parents=True, exist_ok=True)
    return config
