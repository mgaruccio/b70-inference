#!/usr/bin/env python3
"""Fail-closed decoder-marker extension of the completed SYCL analyzer."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCL = HERE.parent / '20261019-lilicorr-b70-sycl-profile'
_CURRENT = runpy.run_path(str(SYCL / 'analyze-profile.py'))
_CURRENT_GLOBALS = _CURRENT['event'].__globals__
_OLD_VALID_MARKER = _CURRENT_GLOBALS['_valid_marker_name']


def _valid_marker_name(name):
    # Keep every completed marker contract and add only the exact draft-decoder
    # input tensor.  Prefixes or target-side lookalikes remain invalid.
    return name == 'inp_noise_embd' or _OLD_VALID_MARKER(name)


# Functions loaded by runpy resolve globals through their actual function
# dictionary, not the returned namespace copy. Replace only its marker
# predicate; all schema, coverage, phase correlation, and restoration checks
# remain the completed implementation.
_CURRENT_GLOBALS['_valid_marker_name'] = _valid_marker_name
BASE = _CURRENT_GLOBALS['BASE']
MARKER = _CURRENT_GLOBALS['MARKER']
for _name in (
    'event', 'parse_sycl', '_graph_bundles', '_phase_owner', '_phase_parent',
    '_scheduler_owners', '_unattributed_sycl', 'request_sycl', '_startup_sycl',
    '_nonmeasured_sycl', 'analyze',
):
    globals()[_name] = _CURRENT_GLOBALS[_name]
CURRENT = _CURRENT_GLOBALS
