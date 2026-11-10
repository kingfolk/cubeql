"""Independent exact-vector adapter verifies the production bundle's text and ordered IDs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/external'))
from reference import load_vectors

@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('blake3'),
                     'install encoder requirements for production-vector reference tests')
class ProductionReferenceTests(unittest.TestCase):
    def test_production_bundle_matches_dataset_and_rejects_reordered_or_changed_content(self):
        import numpy as np
        import blake3
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'dataset'; (dataset/'qrels').mkdir(parents=True)
            (dataset/'corpus.jsonl').write_text('{"_id":"d1","title":"Title","text":"one"}\n{"_id":"d2","text":"two"}\n')
            (dataset/'queries.jsonl').write_text('{"_id":"q","text":"question"}\n')
            (dataset/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq\td1\t1\n')
            def record(identifier,text):
                return {'id':identifier,'text':text,'content_hash':blake3.blake3(text.encode()).hexdigest()}
            inputs={'format_version':1,'documents':[record('d1','Title\none'),record('d2','two')],'queries':[record('q','question')]}
            bundle=root/'bundle';bundle.mkdir()
            (bundle/'documents.f32').write_bytes(np.asarray([[1,0],[0,1]],dtype='<f4').tobytes())
            (bundle/'queries.f32').write_bytes(np.asarray([[1,0]],dtype='<f4').tobytes())
            manifest={'format_version':1,'dtype':'float32_le','config':{'dimensions':2,'normalization':'l2'}}
            def save():
                (bundle/'inputs.json').write_text(json.dumps(inputs))
                for key,name in [('inputs','inputs.json'),('documents','documents.f32'),('queries','queries.f32')]:
                    manifest[key]={'file':name,'sha256':hashlib.sha256((bundle/name).read_bytes()).hexdigest()}
                (bundle/'vectors.json').write_text(json.dumps(manifest))
            save()
            loaded,matrices=load_vectors(dataset,bundle/'vectors.json')
            self.assertEqual(loaded['document_ids'],['d1','d2'])
            self.assertEqual(matrices['documents'].shape,(2,2))
            inputs['documents'].reverse();save()
            with self.assertRaises(ValueError):load_vectors(dataset,bundle/'vectors.json')
            inputs['documents'].reverse()
            inputs['documents'][0]=record('d1','changed source');save()
            with self.assertRaises(ValueError):load_vectors(dataset,bundle/'vectors.json')

if __name__=='__main__':unittest.main()
