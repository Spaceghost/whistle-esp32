"""A checkpoint loader's defaults must not override its training signature."""
from dataclasses import dataclass
from pathlib import Path
import runpy
import unittest

runtime = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/english_runtime.py'))


@dataclass
class Config:
    dim: int = 512
    languages: tuple = ('en',)
    dtype: str = 'float32'
    flash: bool = True
    remat: bool = True
    scan_unroll: int = 1

    @classmethod
    def from_saved(cls, saved):
        return cls(dim=saved['dim'], languages=tuple(saved['languages']))


class Model:
    WhistleConfig = Config


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(dim=512, languages=['en'], flash=False)
        self.signature = {'config':dict(dim=512,languages=['en'],dtype='float32',flash=False,remat=False,scan_unroll=2)}

    def test_restore_options_the_loader_discards(self):
        cfg = runtime['configuration'](Model,self.saved,self.signature)
        self.assertFalse(cfg.flash)
        self.assertFalse(cfg.remat)
        self.assertEqual(cfg.scan_unroll,2)
        self.assertEqual(cfg.languages,('en',))

    def test_missing_option_refused(self):
        del self.signature['config']['flash']
        with self.assertRaisesRegex(ValueError,'missing'):
            runtime['configuration'](Model,self.saved,self.signature)

    def test_geometry_mismatch_refused(self):
        self.signature['config']['dim']=256
        with self.assertRaisesRegex(ValueError,'geometry'):
            runtime['configuration'](Model,self.saved,self.signature)

    def test_malformed_options_refused(self):
        for key,value in (('flash',1),('remat','true'),('scan_unroll',0),('scan_unroll',True),('dtype','bfloat16')):
            with self.subTest(key=key,value=value):
                original=self.signature['config'][key]
                self.signature['config'][key]=value
                with self.assertRaises(ValueError):
                    runtime['configuration'](Model,self.saved,self.signature)
                self.signature['config'][key]=original


if __name__ == '__main__':
    unittest.main()
