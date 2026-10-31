import importlib.util
from pathlib import Path
import struct
import unittest
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
