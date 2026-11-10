"""Verify published benchmark provenance, status, links and figure payloads."""
import argparse
import csv
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from PIL import Image
from .run import ROOT, MAX_OUTPUT_BYTES, code_sha, sha, verify_sources


def verify(directory):
    directory = Path(directory).resolve()
    verify_sources()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['implementation_sha256'].items():
        if code_sha(ROOT/name) != digest: raise ValueError(f'Producer code hash differs: {name}')
    artifacts = json.loads((directory/'artifacts.json').read_text(encoding='utf-8'))['files']
    expected = {directory/'artifacts.json'}
    for entry in artifacts:
        path = directory/entry['path']
        if not path.resolve().is_relative_to(directory) or path.is_symlink() or not path.is_file():
            raise ValueError(f'Invalid artifact path: {entry["path"]}')
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'Artifact bytes changed: {entry["path"]}')
        expected.add(path)
    actual = {p for p in directory.rglob('*') if p.is_file()}
    if actual != expected: raise ValueError('Artifact inventory differs from recorded file set')
    if sum(p.stat().st_size for p in actual)>MAX_OUTPUT_BYTES: raise ValueError('Collection exceeds byte budget')
    records = json.loads((directory/'index.json').read_text(encoding='utf-8'))
    for record in records:
        name = record['name']; case = directory/name
        flag = record['metrics']['converged']
        if flag != record['expected_converged']: raise ValueError(f'Unexpected solve outcome: {name}')
        summary = json.loads((case/'summary.json').read_text(encoding='utf-8'))
        if summary['converged'] != flag or (case/'INCOMPLETE').exists(): raise ValueError(f'Incomplete/mismatched status: {name}')
        for pattern in ('nodes-*.csv','loads-*.csv','terminals-*.csv'):
            for path in case.glob(pattern):
                with path.open(encoding='utf-8',newline='') as stream:
                    if any(row['converged'] != str(int(flag)) for row in csv.DictReader(stream)):
                        raise ValueError(f'CSV status mismatch: {path.name}')
        if 'reference' in record and (not record['reference']['converged'] or
                                     record['reference']['max_complex_voltage_error_pu']>record['reference']['acceptance_tolerance_pu']):
            raise ValueError(f'Reference mismatch: {name}')
    images = 0
    for path in actual:
        if path.suffix == '.png':
            with Image.open(path) as im:
                if im.width>1600 or im.height>1100: raise ValueError(f'Oversized preview: {path}')
                im.verify()
            images += 1
        elif path.suffix == '.svg': ElementTree.parse(path)
        elif path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                if '://' not in target and not target.startswith('#') and not (path.parent/target.split('#')[0]).exists():
                    raise ValueError(f'Broken local link in {path}: {target}')
    return {'studies':len(records),'converged':sum(r['metrics']['converged'] for r in records),
            'expected_nonconvergence':sum(not r['expected_converged'] for r in records),
            'files':len(actual),'png_previews':images,'logical_mib':sum(p.stat().st_size for p in actual)/2**20,
            'source_hashes':'verified','artifact_hashes':'verified','local_links':'verified'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'benchmarks/results')
    print(json.dumps(verify(parser.parse_args().out),indent=2))
