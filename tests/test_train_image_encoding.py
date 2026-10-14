"""Reject incompatible ICL presentation contracts before starting the simulator."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from roboicl.policy.observation_images import image_encoding, validate_train_image_encoding
from roboicl.run import reference_preflight


class TrainImageEncodingTest(unittest.TestCase):
    def test_bundle_native_images_can_be_presented_as_separate_views(self):
        validate_train_image_encoding(image_encoding('native_jpeg', 'original', 'triptych'),
                                      'native_jpeg', 'original', 'separate', bundle=True)

    def test_high_detail_cannot_relabel_original_train_archive(self):
        with self.assertRaisesRegex(ValueError, 'encoding contract'):
            validate_train_image_encoding(image_encoding('native_jpeg', 'original', 'triptych'),
                                          'native_jpeg', 'high', 'triptych', bundle=True)

    def test_preflight_rejects_mismatch_before_reading_source_or_sidecar(self):
        bundle = {'task': 'deposit_coin',
                  'prompt_image_encoding': image_encoding('native_jpeg', 'original', 'triptych')}
        bundle['sha256'] = hashlib.sha256(json.dumps(bundle, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train_reference_bundle.json').write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError, 'encoding contract'):
                reference_preflight('deposit_coin', 1,
                    {'image_profile': 'native_jpeg', 'image_detail': 'high', 'image_layout': 'triptych'},
                    root, {}, root)


if __name__ == '__main__':
    unittest.main()
