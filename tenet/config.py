"""YAML configs with a `base:` key for inheritance and `--key value` command-line overrides."""

import os
import argparse
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('base', None)
    if base is not None:
        merged = load_config(os.path.join(os.path.dirname(path), base))
        merged.update(cfg)
        cfg = merged
    return cfg


def _str2bool(v):
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'boolean value expected, got {v}')


def add_override_arguments(parser, cfg):
    """Expose every config key as an optional `--key value` argument."""
    for key, value in cfg.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=_str2bool, default=None)
        elif isinstance(value, (int, float, str)) or value is None:
            kind = type(value) if value is not None else str
            parser.add_argument(f'--{key}', type=kind, default=None)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
