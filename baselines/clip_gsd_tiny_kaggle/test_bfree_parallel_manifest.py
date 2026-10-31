"""Parallel header reading must produce byte-identical manifest/identity files to the original."""
import ast, hashlib, io, json, struct, tempfile, time, unittest, zlib
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import build_bfree_samecat_origbg_notebooks as b

HERE = Path(__file__).resolve().parent


def pick(ns, file, names):
    tree = ast.parse((HERE / file).read_text(encoding='utf-8'))
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), file, 'exec'), ns)


class ParallelManifestTest(unittest.TestCase):
    def test_identical_outputs(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / 'bfree'
        ids = [f'{200000000000 + i:012d}' for i in range(30)]
        valid = ids[:12]
        root.mkdir()
        pd.DataFrame(dict(id=ids, same_category='a', diff_category='b')).to_csv(root / 'train_list.csv', index=False)
        pd.DataFrame(dict(id=valid, same_category='a', diff_category='b')).to_csv(root / 'valid_list.csv', index=False)
        ns = dict(hashlib=hashlib, json=json, struct=struct, io=io, pd=pd, Path=Path, time=time)
        pick(ns, 'runtime.py', {'digest'})
        pick(ns, 'evaluate_uploaded_best.py', {'atomic_json', 'bind_upload_metadata'})
        pick(ns, 'evaluate_all7_variants.py', {'save_csv'})
        pick(ns, 'evaluate_bfree_valid.py', {'BFREE_REAL', 'BFREE_BRANCHES', 'bfree_list', 'bfree_folder',
                                             'bfree_png_header', 'prepare_bfree_manifest'})
        ns.update(BFREE_IMAGE_SIDE=8, BFREE_ALL_COUNT=30, BFREE_VALID_COUNT=12, BFREE_LIST_MD5=dict(
            train_list=hashlib.md5((root / 'train_list.csv').read_bytes()).hexdigest(),
            valid_list=hashlib.md5((root / 'valid_list.csv').read_bytes()).hexdigest()))
        for k, name in enumerate((ns['BFREE_REAL'], *ns['BFREE_BRANCHES'])):
            folder = root / name / name if k % 2 else root / name
            folder.mkdir(parents=True)
            for i in valid:
                Image.fromarray(np.full((8, 8, 3), k * 10 + int(i[-2:]), np.uint8)).save(folder / f'{i}.png')
        opts = dict(smoke_only=False, header_log_every=20)
        out_a, out_b = Path(tmp.name) / 'a', Path(tmp.name) / 'b'
        out_a.mkdir(); out_b.mkdir()
        m1, ids1, id1 = ns['prepare_bfree_manifest'](root, opts, out_a)
        exec(b.PARALLEL_MANIFEST_CELL, ns)
        self.assertIsNot(ns['prepare_bfree_manifest'], ns['_original_prepare_bfree_manifest'])
        m2, ids2, id2 = ns['prepare_bfree_manifest'](root, opts, out_b)
        self.assertEqual(ids1, ids2)
        self.assertEqual(id1, id2)
        pd.testing.assert_frame_equal(m1, m2)
        for name in ('manifest_identity.json', 'bfree_valid_manifest.csv', 'bfree_valid_ids.csv', 'bfree_manifest_summary.json'):
            self.assertEqual((out_a / name).read_bytes(), (out_b / name).read_bytes(), name)
        name = ns['BFREE_BRANCHES'][2]
        (root / name / name / f'{valid[3]}.png').unlink()
        with self.assertRaises(FileNotFoundError):
            ns['prepare_bfree_manifest'](root, opts, Path(tmp.name) / 'c')
        tmp.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)
