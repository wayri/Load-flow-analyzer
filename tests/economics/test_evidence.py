"""Published inputs/results retain exact provenance and independently replay."""
import hashlib
import json
from pathlib import Path
import unittest
from nrlarge.economics.grid import check_grid
from nrlarge.economics.io import load_case, load_dispatch
from nrlarge.economics.management import baseline_case
from nrlarge.economics.validation import validate_dispatch


ROOT=Path(__file__).resolve().parents[2]
GALLERY=ROOT/'economics/examples'


class PublishedEvidence(unittest.TestCase):
    def test_exact_artifact_hashes_and_canonical_producer_hashes(self):
        manifest=json.loads((GALLERY/'manifest.json').read_text(encoding='utf-8'))
        self.assertLess(sum(v['bytes'] for v in manifest['artifacts'].values()),4*2**20)
        for name,record in manifest['artifacts'].items():
            with self.subTest(artifact=name):
                data=(GALLERY/name).read_bytes()
                self.assertEqual(len(data),record['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
        for name,digest in manifest['producer_sha256_lf'].items():
            data=(ROOT/'nrlarge/economics'/name).read_bytes().replace(b'\r\n',b'\n')
            self.assertEqual(hashlib.sha256(data).hexdigest(),digest,name)

    def test_all_role_replan_and_sizing_receipts_replay(self):
        folders=[GALLERY/role for role in ('battery','generator','discom','aggregator','replan')]
        folders+=sorted((GALLERY/'sizing').glob('candidate-*'))
        for folder in folders:
            with self.subTest(folder=folder.name):
                case=load_case(folder/'inputs'); result=load_dispatch(folder)
                self.assertTrue(result.feasible); self.assertTrue(result.optimal)
                report=validate_dispatch(case,result)
                self.assertTrue(report['valid'],report['violations'])
                if (folder/'baseline-dispatch.json').exists():
                    baseline=load_dispatch(folder/'baseline-dispatch.json')
                    self.assertTrue(validate_dispatch(baseline_case(case),baseline)['valid'])

    def test_ac_gallery_has_configured_pass_and_strict_loss_detection(self):
        case=load_case(GALLERY/'battery/inputs'); result=load_dispatch(GALLERY/'battery')
        folder=GALLERY/'ac-replay'
        for name,passed in (('mapping.json',True),('mapping-strict.json',False)):
            mapping=json.loads((folder/name).read_text(encoding='utf-8'))
            report=check_grid(case,result,mapping,folder)
            self.assertEqual(report['passed'],passed)
            self.assertTrue(all(r['converged'] for r in report['intervals']))


if __name__=='__main__':
    unittest.main()
