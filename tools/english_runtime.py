"""Restore recorded training execution options discarded by from_saved()."""
from dataclasses import asdict, replace
import json

RUNTIME_FIELDS = ('dtype', 'flash', 'remat', 'scan_unroll')


def configuration(whistle, saved, signature):
    base = whistle.WhistleConfig.from_saved(saved)
    recorded = signature['config']
    canonical = lambda value: json.dumps(value, sort_keys=True)
    for key, value in asdict(base).items():
        if key not in RUNTIME_FIELDS and (key not in recorded or canonical(recorded[key]) != canonical(value)):
            raise ValueError('Checkpoint geometry differs from training signature: '+key)
    if any(key not in recorded for key in RUNTIME_FIELDS):
        raise ValueError('Training runtime settings are missing')
    if recorded['dtype'] != 'float32' or type(recorded['flash']) is not bool or type(recorded['remat']) is not bool:
        raise ValueError('Unsupported or malformed training execution settings')
    if type(recorded['scan_unroll']) is not int or recorded['scan_unroll'] < 1:
        raise ValueError('Invalid recorded scan unroll')
    return replace(base, **{key: recorded[key] for key in RUNTIME_FIELDS})


def describe(cfg):
    return {key: getattr(cfg, key) for key in RUNTIME_FIELDS}
