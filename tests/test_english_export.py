import importlib.util
from pathlib import Path
import struct
import unittest
from copy import deepcopy
spec=importlib.util.spec_from_file_location('english_export',Path(__file__).parents[1]/'tools/export_english_student.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def fixture(layers=2,sites=1,version=1):
    index=1+layers*27+9+2+sites*4+1
    h=[0]*48+[100000.];h[0]=0x05E12A84;h[1]=index+1;h[2]=28;h[10]=layers;h[31]=sites
    raw=bytearray(struct.pack('<48If',*h)+bytes(112+(index+1)*44))
    offset=len(raw);raw+=struct.pack('<13f',3246,version,2048,8,4,9,128,3,80,16000,512,400,160)
    struct.pack_into('<BBHIIIIQQII',raw,196+112+index*44,2,1,0,13,0,0,0,offset,52,0,0)
    return raw,offset


class EnglishExportTest(unittest.TestCase):
    def checkpoint(self, step=28539):
        config = {'languages': ['en'], 'text': 2048, 'dim': 512}
        return {'metadata': {'step': step, 'initialization': 'fresh random, no inherited model weights',
                             'all_parameters_trainable': True, 'config': config,
                             'signature': {'config': deepcopy(config)}}}

    def test_metadata_accepts_actual_step_instead_of_assuming_two(self):
        for step in (1, 2, 3, 28539):
            self.assertEqual(mod.research_metadata(self.checkpoint(step))['step'], step)

    def test_invalid_checkpoint_step_is_refused(self):
        for step in (0, -1, True, 3.0, '3', None):
            with self.assertRaises(ValueError): mod.research_metadata(self.checkpoint(step))

    def test_checkpoint_signature_geometry_drift_is_refused(self):
        m = self.checkpoint(); m['metadata']['signature']['config']['dim'] = 256
        with self.assertRaises(ValueError): mod.research_metadata(m)

    def test_foreign_or_unexpected_vocabulary_is_refused(self):
        for config in ({'languages': ['en', 'ja'], 'text': 2048},
                       {'languages': ['ja'], 'text': 2048},
                       {'languages': ['en'], 'text': 32000}):
            m = self.checkpoint(); m['metadata']['config'] = config
            m['metadata']['signature']['config'] = deepcopy(config)
            with self.assertRaises(ValueError): mod.research_metadata(m)

    def test_unproven_full_parameter_initialization_is_refused(self):
        for key, value in [('all_parameters_trainable', 1), ('all_parameters_trainable', False),
                           ('initialization', 'inherited')]:
            m = self.checkpoint(); m['metadata'][key] = value
            with self.assertRaises(ValueError): mod.research_metadata(m)

    def test_explicit_profile_changes_only_tag_and_version_words(self):
        for layers in (2,4,8):
            for sites in (0,1,2):
                raw,offset=fixture(layers,sites)
                actual=mod.mark_training_frontend(raw)
                expected=bytearray(raw);struct.pack_into('<I',expected,0,0x31455741);struct.pack_into('<f',expected,offset+4,2.)
                self.assertEqual(actual,expected)
                self.assertEqual(struct.unpack_from('<f',raw,offset+4),(1.,))

    def test_unknown_or_repeated_profile_is_refused(self):
        for version in (0,2,3,float('nan')):
            raw,_=fixture(version=version)
            with self.assertRaises(ValueError):mod.mark_training_frontend(raw)

    def test_stable_profile_three_is_explicit(self):
        raw,offset=fixture()
        actual=mod.mark_training_frontend(raw,3)
        self.assertEqual(struct.unpack_from('<I',actual),(0x31455741,))
        self.assertEqual(struct.unpack_from('<f',actual,offset+4),(3.,))
        for profile in (0,1,4):
            with self.assertRaises(ValueError):mod.mark_training_frontend(raw,profile)

    def test_truncated_manifest_is_refused(self):
        raw,_=fixture()
        with self.assertRaises(ValueError):mod.mark_training_frontend(raw[:-1])


if __name__=='__main__':unittest.main()
