"""Shared, checksum-validated configuration for production PREP and RUN."""
from pathlib import Path
import hashlib
import json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def master_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Manifest paths must be relative to the master folder.')
    resolved = (Path(root)/path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('Manifest path escapes the master folder.')
    return resolved


def load_run_manifest(path, *, stage='prep', master_root=None):
    """Both stages load identical inputs; RUN additionally requires a PREP receipt.

    This validates configuration, not geometry/routing numerical acceptance.
    A PREP implementation must record the manifest hash and hashes of every
    generated preparation file before RUN may consume it.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('schema_version') != 'rapid-master-run-v1' or stage not in ('prep', 'run'):
        raise ValueError('Unknown manifest schema or stage.')
    root = Path(master_root or data['master_root'])
    for relative, expected in data['input_sha256'].items():
        if digest(master_path(root, relative)) != expected:
            raise ValueError('Input changed: '+relative)
    parameters = json.loads(master_path(root, data['calibration_parameters']).read_text())
    if set(parameters) != set(data['examples']) or '11' in parameters:
        raise ValueError('Calibration examples differ from the selected run scope.')
    for ex, row in parameters.items():
        if not .5 <= row['cunge']['beta'] <= 2:
            raise ValueError('Beta outside accepted bounds: '+ex)
    if data['time_policy']['report_interval_seconds'] != 60:
        raise ValueError('Unreviewed reporting interval.')
    if stage == 'run':
        receipt_path = master_path(root, data['outputs']['prep_receipt'])
        if not receipt_path.exists():
            raise ValueError('PREP is not complete; run receipt is missing.')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('manifest_sha256') != digest(path) or not receipt.get('complete'):
            raise ValueError('PREP receipt does not match this run manifest.')
        if not receipt.get('files'):
            raise ValueError('PREP receipt contains no prepared files.')
        for relative, expected in receipt['files'].items():
            if digest(master_path(root, relative)) != expected:
                raise ValueError('Prepared input changed: '+relative)
    return data, parameters
