"""Offline checks for shared foreground geometry and evaluator/CLI integration."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

import calculate_metrics
from evaluator.evaluator import Evaluator
from evaluator.metrics.crop import ForegroundCrops, crop_metric, foreground_bbox
from evaluator.register import get_metrics, initialize_metrics


class CropMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        initialize_metrics()
        cls.metrics = get_metrics()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.num_threads)

    def test_bbox_includes_disconnected_foreground_holes_and_boundary_pixels(self):
        mask = torch.zeros(3, 24, 40)
        mask[0, 3, 8] = 0.2
        mask[0, 23, 39] = 1
        mask[1, 0, 0] = 1  # Match other regional metrics' first-channel convention.
        self.assertEqual(foreground_bbox(mask), (3, 8, 24, 40))
        prediction, target = torch.ones(3, 24, 40), torch.zeros(3, 24, 40)
        self.assertEqual(self.metrics['MSE-CROP'](prediction, target, mask=mask), 1.)
        self.assertLess(self.metrics['MSE-FG'](prediction, target, mask=mask), 0.01)

    def test_real_metrics_match_manual_crops_with_variable_shapes(self):
        generator = torch.Generator().manual_seed(19)
        predictions = [torch.rand(3, 40, 64, generator=generator), torch.rand(3, 72, 48, generator=generator)]
        references = [image * 0.7 for image in predictions]
        masks = [torch.zeros(1, *image.shape[-2:]) for image in predictions]
        masks[0][:, 8:32, 9:58] = 1
        masks[1][:, 12:64, 2:33] = 1
        source_crops = [predictions[0][:, 8:32, 9:58], predictions[1][:, 12:64, 2:33]]
        target_crops = [references[0][:, 8:32, 9:58], references[1][:, 12:64, 2:33]]
        for name in ('MSE', 'PSNR', 'SSIM'):
            with self.subTest(metric=name):
                expected = self.metrics[name](source_crops, target_crops)
                actual = self.metrics[name + '-CROP'](predictions, references, mask=masks)
                self.assertAlmostEqual(actual, expected, places=6)

    def test_prediction_and_reference_share_mask_resolution_and_aspect(self):
        mask = torch.zeros(1, 30, 50)
        mask[:, 5:20, 9:40] = 1
        sources = torch.rand(1, 3, 60, 100)
        targets = torch.rand(1, 3, 45, 75)
        seen = []

        def metric(source, target, **kwargs):
            seen.append((source[0], target[0]))
            return 0.

        crop_metric(metric)(sources, targets, mask=mask)
        self.assertEqual(seen[0][0].shape, (3, 15, 31))
        self.assertEqual(seen[0][0].shape, seen[0][1].shape)
        for original, actual in zip((sources, targets), seen[0]):
            expected = F.interpolate(original, size=(30, 50), mode='bilinear', align_corners=False, antialias=True)
            torch.testing.assert_close(actual, expected[0, :, 5:20, 9:40])

    def test_single_pixel_full_and_empty_masks(self):
        image = torch.rand(1, 3, 12, 20)
        mask = torch.zeros(1, 1, 12, 20)
        mask[..., -1, -1] = 1
        self.assertEqual(foreground_bbox(mask[0]), (11, 19, 12, 20))
        self.assertEqual(self.metrics['MSE-CROP'](image, image, mask=mask), 0.)
        mask.fill_(1)
        self.assertEqual(foreground_bbox(mask[0]), (0, 0, 12, 20))
        self.assertTrue(math.isinf(self.metrics['PSNR-CROP'](image, image, mask=mask)))
        mask.zero_()
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.metrics['MSE-CROP'](image, image, mask=mask)
        with self.assertRaisesRegex(ValueError, 'require a mask'):
            self.metrics['MSE-CROP'](image, image)

    def test_crop_wrapper_keeps_distribution_aggregation_and_prompt_order(self):
        calls = []
        images = [torch.rand(3, 20, 30), torch.rand(3, 30, 20)]
        masks = [torch.ones(1, *image.shape[-2:]) for image in images]
        prompts = ['first', 'second']

        def metric(source, target=None, **kwargs):
            calls.append((source, target, kwargs))
            return float(len(source))

        wrapped = crop_metric(metric)
        self.assertEqual(wrapped(images, images, mask=masks), 2.)
        self.assertEqual(len(calls), 1)  # FID must see the complete distribution once.
        self.assertEqual(wrapped(images, mask=masks, prompts=prompts), 2.)
        self.assertIsNone(calls[1][1])
        self.assertIs(calls[1][2]['prompts'], prompts)
        self.assertNotIn('mask', calls[1][2])

    def test_views_are_lazy_and_reuse_shared_bounds(self):
        class Images:
            calls = []

            def __len__(self):
                return 1

            def __getitem__(self, index):
                self.calls.append(index)
                return torch.zeros(3, 20, 30)

        images = Images()
        crops = ForegroundCrops(images, [((20, 30), (1, 2, 10, 25))])
        self.assertEqual(images.calls, [])
        self.assertEqual(crops[0].shape, (3, 9, 23))
        self.assertEqual(images.calls, [0])

    def test_registration_is_complete_and_idempotent(self):
        initialize_metrics()
        self.assertEqual(get_metrics(), self.metrics)
        for name in ('MSE', 'PSNR', 'SSIM', 'CLIP', 'DINO', 'FID', 'LPIPS', 'DISTS', 'VGG-CONTENT', 'CLIP-TEXT'):
            self.assertIn(name + '-CROP', self.metrics)
        self.assertFalse(any(name.endswith(('-FG-CROP', '-CROP-CROP')) for name in self.metrics))

    def test_evaluator_supports_mixed_whole_and_crop_metrics(self):
        prediction, reference = torch.ones(1, 3, 24, 40), torch.zeros(1, 3, 24, 40)
        prediction[..., :12, :] = 0
        mask = torch.zeros(1, 1, 24, 40)
        mask[..., 12:, :] = 1
        evaluator = Evaluator(['MSE', 'MSE-CROP'])
        self.assertEqual(evaluator.compute(prediction, reference, mask=mask), {'MSE': .5, 'MSE-CROP': 1.})
        self.assertEqual(evaluator.compute(prediction, reference), {'MSE': .5})
        evaluator = Evaluator(['CLIP-TEXT-CROP'])
        with patch('evaluator.metrics.clip_text._load_model') as load:
            # Missing text is caught before attempting model loading.
            with self.assertRaisesRegex(ValueError, 'requires prompts'):
                evaluator.compute(prediction, mask=mask)
            load.assert_not_called()

    def test_cli_supports_crop_region_explicit_names_and_reports_empty_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ('source', 'target', 'mask', 'predictions'):
                (root / role).mkdir()
            for index in range(2):
                Image.new('RGB', (40, 24), 'black').save(root / 'source' / f'{index}.png')
                Image.new('RGB', (40, 24), 'black').save(root / 'target' / f'{index}.png')
                prediction = Image.new('RGB', (40, 24), 'black')
                prediction.paste('white', (0, 12, 40, 24))
                prediction.save(root / 'predictions' / f'{index}.png')
                mask = Image.new('L', (40, 24))
                mask.paste(255, (0, 12, 40, 24))
                mask.save(root / 'mask' / f'{index}.png')
            manifest = root / 'data.jsonl'
            manifest.write_text(''.join(json.dumps({
                'target': f'target/{index}.png', 'prompt': f'edit {index}',
                'conditions': {'source': f'source/{index}.png', 'mask': f'mask/{index}.png'},
            }) + '\n' for index in range(2)))
            arguments = ['calculate_metrics.py', '--data-file', str(manifest), '--pred-dir', str(root / 'predictions'),
                         '--preprocess', 'resize', '--device', 'cpu', '--output', str(root / 'metrics.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                for flags, expected in ((['--region', 'crop', '--metrics', 'MSE'], {'MSE-CROP': 1.}),
                                        (['--metrics', 'MSE', 'MSE-CROP'], {'MSE': .5, 'MSE-CROP': 1.})):
                    with patch('sys.argv', arguments + flags):
                        self.assertEqual(calculate_metrics.main(), 0)
                    report = json.loads((root / 'metrics.json').read_text())
                    self.assertEqual(report['metrics'], expected)
                    self.assertEqual(report['failed_metrics'], {})
                    self.assertEqual(report['skipped_metrics'], {})
                def text_score(images, reference=None, prompts=None, **kwargs):
                    self.assertIsNone(reference)
                    self.assertEqual(prompts, ['edit 0', 'edit 1'])
                    self.assertEqual(images[0].shape, (3, 12, 40))
                    return .75

                with (
                    patch.object(calculate_metrics, 'initialize_metrics'),
                    patch.object(calculate_metrics, 'get_metrics', return_value={
                        'CLIP-TEXT-CROP': crop_metric(text_score),
                    }),
                    patch('sys.argv', arguments + ['--metrics', 'CLIP-TEXT-CROP', '--clip-model-id', str(root)]),
                ):
                    self.assertEqual(calculate_metrics.main(), 0)
                report = json.loads((root / 'metrics.json').read_text())
                self.assertEqual(report['metrics'], {'CLIP-TEXT-CROP': .75})
                self.assertEqual(report['metric_references'], {'CLIP-TEXT-CROP': 'prompt'})
                Image.new('L', (40, 24)).save(root / 'mask' / '0.png')
                with patch('sys.argv', arguments + ['--metrics', 'MSE', 'MSE-CROP']):
                    self.assertEqual(calculate_metrics.main(), 0)
                report = json.loads((root / 'metrics.json').read_text())
                self.assertEqual(report['metrics'], {'MSE': .5})
                self.assertIn('empty crop', report['skipped_metrics']['MSE-CROP'])


if __name__ == '__main__':
    unittest.main()
