"""CPU math and training fixtures; these do not measure UCF-Crime performance."""
import contextlib
import copy
import importlib
import io
import json
import math
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.msc_branch.config import training_parser, validate_args
from experiments.msc_branch.data import source_video_id, VideoIdentityDataset
from experiments.msc_branch.representation import RepresentationBranch, calibrate_center, contrastive_loss
from experiments.msc_branch.checkpoint import capture_rng, restore_rng, load_checkpoint

torch.set_num_threads(1)


class LossTests(unittest.TestCase):
    def inputs(self):
        return (torch.tensor([[[0., 1.]], [[1., 0.]], [[1., 0.]]]), torch.zeros(3, 1, 1),
                torch.ones(3, dtype=torch.long), torch.tensor([0, 1, 1]), ['normal', 'abuseA', 'abuseB'])

    def test_numeric_loss_and_gradient_not_score_or_center(self):
        z, c, lengths, labels, ids = self.inputs()
        z.requires_grad_(); c.requires_grad_()
        center = torch.zeros(2, requires_grad=True)
        loss, stats = contrastive_loss(z, c, lengths, labels, ids, 'mean_shifted', center, 1.)
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-1)), places=6)
        self.assertEqual(stats['anchors_used'], 2)
        loss.backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertIsNone(c.grad)
        self.assertIsNone(center.grad)

    def test_repeating_crops_does_not_reweight_sources(self):
        z, c, lengths, labels, ids = self.inputs()
        z[2] = torch.tensor([[.6, .8]])
        base, _ = contrastive_loss(z, c, lengths, labels, ids, temperature=1.)
        for selected in ([0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 0, 1, 2]):
            repeated, _ = contrastive_loss(z[selected], c[selected], lengths[selected], labels[selected],
                                           [ids[i] for i in selected], temperature=1.)
            torch.testing.assert_close(base, repeated)

    def test_same_source_no_partner_no_normal_returns_differentiable_zero(self):
        z, c, lengths, labels, ids = self.inputs()
        for changed_labels, changed_ids in [(labels, ['normal', 'same', 'same']),
                                            (torch.tensor([0, 1, 2]), ids),
                                            (torch.tensor([3, 1, 1]), ids)]:
            values = z.clone().requires_grad_()
            loss, stats = contrastive_loss(values, c, lengths, changed_labels, changed_ids)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats['anchors_used'], 0)
            loss.backward()
            self.assertEqual(float(values.grad.abs().sum()), 0)

    def test_other_event_is_excluded_and_padding_is_masked(self):
        z, c, lengths, labels, ids = self.inputs()
        extra = torch.tensor([[[.3, .7]]])
        z = torch.cat((z, extra)).repeat(1, 2, 1)
        z[:, 1] = float('nan')
        z.requires_grad_()
        c = torch.zeros(4, 2, 1); c[:, 1] = float('nan')
        loss, _ = contrastive_loss(z, c, torch.ones(4), torch.tensor([0, 1, 1, 2]), ids+['arson'])
        original, _ = contrastive_loss(*self.inputs())
        torch.testing.assert_close(loss, original)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[:, 1].abs().sum()), 0)
        self.assertEqual(float(z.grad[3].abs().sum()), 0)

    def test_mean_shift_matches_manual_transform_and_degenerate_center_is_safe(self):
        z, c, lengths, labels, ids = self.inputs()
        center = torch.tensor([.2, .3])
        shifted, _ = contrastive_loss(z, c, lengths, labels, ids, 'mean_shifted', center)
        manual = torch.nn.functional.normalize(z - center, dim=-1)
        expected, _ = contrastive_loss(manual, c, lengths, labels, ids)
        torch.testing.assert_close(shifted, expected)
        values = z.clone().requires_grad_()
        loss, stats = contrastive_loss(values, c, lengths, labels, ids, 'mean_shifted', torch.tensor([1., 0.]))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats['near_center_skipped'], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_unequal_lengths_have_equal_key_weight_per_source(self):
        # Source A has K=2 identical keys, B has K=1: same result as one key each.
        z, c, lengths, labels, ids = self.inputs()
        full = z.repeat(1, 17, 1)
        actual, _ = contrastive_loss(full, torch.zeros(3, 17), torch.tensor([17, 1, 17]), labels, ids)
        expected, _ = contrastive_loss(z, c, lengths, labels, ids)
        torch.testing.assert_close(actual, expected)

    def test_invalid_inputs_and_small_temperature(self):
        z, c, lengths, labels, ids = self.inputs()
        for kwargs in ({'temperature': 0}, {'temperature': float('nan')}, {'mode': 'none'},
                       {'mode': 'mean_shifted', 'center': torch.zeros(3)}):
            with self.assertRaises(ValueError): contrastive_loss(z, c, lengths, labels, ids, **kwargs)
        for bad_lengths in (torch.tensor([0, 1, 1]), torch.tensor([1, 2, 1]), torch.tensor([1., 1., .5])):
            with self.assertRaises(ValueError): contrastive_loss(z, c, bad_lengths, labels, ids)
        value, _ = contrastive_loss(z, c, lengths, labels, ids, temperature=1e-5)
        self.assertTrue(torch.isfinite(value))


class CalibrationAndDataTests(unittest.TestCase):
    def test_source_identity_and_projector_rng_padding(self):
        self.assertEqual(source_video_id(r'C:\features\Abuse001_x264__0.npy', 'Abuse'),
                         source_video_id('/data/Abuse001_x264__9.npy', 'Abuse'))
        self.assertNotEqual(source_video_id('Abuse001.npy', 'Abuse'), source_video_id('Abuse002.npy', 'Abuse'))
        self.assertEqual(source_video_id('anything.npy', 'Abuse', 'source_a'), 'source_a')
        torch.manual_seed(11)
        before = torch.get_rng_state()
        branch = RepresentationBranch(4, 3, 2)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        h = torch.randn(2, 3, 4); h[1, 1:] = float('nan'); h.requires_grad_()
        z = branch(h, torch.tensor([3, 1]))
        self.assertTrue(torch.isfinite(z).all())
        z.sum().backward()
        self.assertEqual(float(h.grad[1, 1:].abs().sum()), 0)

    def test_calibration_train_only_equal_video_mean_and_rng(self):
        class Dataset:
            normal, test_mode = True, False
            source_ids = ['n1', 'n1', 'n2']
            def __getitem__(self, index):
                torch.rand(1); np.random.rand(); random.random()
                return torch.tensor([[1., 0.], [0., 1.]]), 'Normal', 2, self.source_ids[index]
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.representation = RepresentationBranch(2, 2, 2)
                self.encoder = nn.Identity()
            def encode_video(self, x, *args): return self.encoder(x)
        model, dataset = Model(), Dataset()
        model.train(); model.encoder.eval()
        args = training_parser().parse_args(['--output-dir', 'unused'])
        rng = capture_rng()
        expected_center = model.representation(torch.tensor([[[1., 0.], [0., 1.]]]), [2]).mean(dim=(0, 1)).detach()
        info = calibrate_center(model, dataset, args, 'cpu', 3)
        torch.testing.assert_close(model.representation.center, expected_center)
        self.assertEqual(info['videos'], 2)
        self.assertEqual(info['samples'], 4)
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.training)
        expected = (torch.rand(2), np.random.rand(), random.random())
        restore_rng(rng)
        actual = (torch.rand(2), np.random.rand(), random.random())
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        self.assertEqual(expected[1:], actual[1:])
        dataset.test_mode = True
        with self.assertRaises(ValueError): calibrate_center(model, dataset, args, 'cpu', 3)


class ModelAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.cnn_memory.tests.test_cnn_memory import FakeCLIP, tokenize
        fake = types.ModuleType('clip')
        fake.clip = types.SimpleNamespace(load=lambda *args: (FakeCLIP(), None), tokenize=tokenize)
        cls.clip_patch = patch.dict(sys.modules, {'clip': fake})
        cls.clip_patch.start()
        cls.module = importlib.import_module('experiments.msc_branch.msc_model')

    @classmethod
    def tearDownClass(cls): cls.clip_patch.stop()

    def make_model(self):
        torch.manual_seed(234)
        model = self.module.MSCBranchCLIPVAD(3, 8, 8, 8, 1, 1, 4, 1, 1, 'cpu',
                    feature_dim=6, memory_slots=2, representation_hidden=6, representation_dim=4)
        model.disAdj.forward = lambda batch, length: torch.eye(length).expand(batch, -1, -1)
        return model.eval()

    def test_outputs_match_cnn_wrapper_inference_skips_mlp_and_gradient_reaches_lgt(self):
        model = self.make_model()
        inputs = torch.randn(4, 8, 8), None, ['normal', 'abuse', 'arson'], torch.tensor([8]*4)
        for active in (False, True):
            model.set_memory_active(active)
            expected = self.module.CNNMemoryCLIPVAD.forward(model, *inputs)
            with patch.object(model.representation, 'forward', side_effect=AssertionError('Inference ran MLP')):
                actual = model(*inputs)
            for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
        _, c, _, auxiliary = model(*inputs, return_aux=True, return_representation=True)
        loss, _ = contrastive_loss(auxiliary['representation'], c, inputs[3], torch.tensor([0, 1, 1, 2]),
                                   ['n', 'a', 'b', 'c'])
        loss.backward()
        for parameter in (model.linear.weight, model.representation.projector[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(model.classifier.weight.grad)
        self.assertIsNone(model.memory.prototypes.grad)

    def test_initialization_preserves_cnn_weights_and_rng(self):
        torch.manual_seed(234)
        original = self.module.CNNMemoryCLIPVAD(3, 8, 8, 8, 1, 1, 4, 1, 1, 'cpu', feature_dim=6, memory_slots=2)
        rng = torch.get_rng_state()
        new = self.make_model()
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

    def test_training_modes_resume_and_warmup(self):
        from experiments.msc_branch import train as entrypoint
        from experiments.cnn_memory.cnn_memory import CNNVisualMemory
        class TinyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(512, 8)
                self.classifier = nn.Linear(8, 1)
                self.text = nn.Parameter(torch.randn(14, 8))
                self.memory = CNNVisualMemory(8, 6, 14, slots=2)
                self.representation = RepresentationBranch(8, 6, 4)
                self.register_buffer('memory_active', torch.tensor(False))
            def encode_video(self, visual, *args): return self.encoder(visual.float())
            def set_memory_active(self, active): self.memory_active.fill_(active)
            def forward(self, visual, mask, text, lengths, return_aux=False, return_representation=False):
                h = self.encode_video(visual)
                c, a = self.classifier(h), h @ self.text.T
                auxiliary = self.memory(h, lengths) if bool(self.memory_active) else {}
                if auxiliary:
                    c, a = c + .1*auxiliary['delta_c'], a + .1*auxiliary['delta_a']
                if return_representation: auxiliary['representation'] = self.representation(h, lengths)
                outputs = self.text, c, a
                return (*outputs, auxiliary) if return_aux else outputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, label in enumerate(['Normal', 'Normal', 'Abuse', 'Abuse']):
                path = root / f'{label}{index}__0.npy'
                np.save(path, np.random.default_rng(index).normal(size=(5, 512)).astype('float32'))
                rows.append(f'{path},{label}')
            csv = root/'data.csv'; csv.write_text('path,label\n'+'\n'.join(rows)+'\n')
            gt, segments, labels = root/'gt.npy', root/'segments.npy', root/'labels.npy'
            np.save(gt, np.repeat([0]*10+[1]*10, 16))
            np.save(segments, np.zeros((4, 1, 2)))
            np.save(labels, np.array(['Normal', 'Normal', 'Abuse', 'Abuse']))
            common = ['--train-list', str(csv), '--test-list', str(csv), '--gt-path', str(gt),
                      '--gt-segment-path', str(segments), '--gt-label-path', str(labels),
                      '--batch-size', '2', '--visual-length', '8', '--attn-window', '4',
                      '--memory-start-epoch', '1', '--contrastive-start-epoch', '2', '--eval-steps', '1',
                      '--feature-dim', '6', '--memory-slots', '2', '--representation-hidden', '6',
                      '--representation-dim', '4']
            def run(name, mode, epochs=3, resume=False, extra=()):
                output = root/name
                flags = ['train.py', *common, '--output-dir', str(output), '--contrastive-mode', mode,
                         '--max-epoch', str(epochs), *extra]
                if resume: flags += ['--resume', str(output/'last.pt')]
                with patch.object(sys, 'argv', flags), patch.object(entrypoint, 'training_device', return_value='cpu'), \
                     patch.object(self.module, 'build_model', side_effect=lambda *args: TinyDetector()), \
                     patch('experiments.cnn_memory.evaluation.getDetectionMAP', return_value=([0.]*5, [.1,.2,.3,.4,.5])), \
                     contextlib.redirect_stdout(io.StringIO()):
                    entrypoint.main()
                return load_checkpoint(output/'last.pt')
            arms = {mode: run(mode, mode) for mode in ('none', 'standard', 'mean_shifted')}
            # Verify the new trainer with loss disabled reproduces the existing
            # CNN trainer's entire trajectory on the same CPU fixture.
            from experiments.cnn_memory import train as old_entrypoint
            from experiments.cnn_memory import cnn_memory_model
            old_common = []
            for index in range(0, len(common), 2):
                if common[index] not in ('--contrastive-start-epoch', '--representation-hidden', '--representation-dim'):
                    old_common += common[index:index+2]
            legacy_output = root/'legacy_cnn'
            with patch.object(sys, 'argv', ['train.py', *old_common, '--output-dir', str(legacy_output), '--max-epoch', '3']), \
                 patch.object(old_entrypoint, 'training_device', return_value='cpu'), \
                 patch.object(cnn_memory_model, 'build_model', side_effect=lambda *args: TinyDetector()), \
                 patch('experiments.cnn_memory.evaluation.getDetectionMAP', return_value=([0.]*5, [.1,.2,.3,.4,.5])), \
                 contextlib.redirect_stdout(io.StringIO()):
                old_entrypoint.main()
            legacy = torch.load(legacy_output/'last.pt', weights_only=True)
            for key, value in arms['none']['model_state_dict'].items():
                torch.testing.assert_close(value, legacy['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual((legacy_output/'metrics.jsonl').read_text(), (root/'none/metrics.jsonl').read_text())
            run('resumed', 'mean_shifted', 2)
            resumed = run('resumed', 'mean_shifted', 3, True)
            self.assertEqual(resumed['best_metrics'], arms['mean_shifted']['best_metrics'])
            for key, value in resumed['model_state_dict'].items():
                torch.testing.assert_close(value, arms['mean_shifted']['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual((root/'resumed/train.log').read_text().count('center_calibrated='), 1)
            for mode in arms:
                log = (root/mode/'train.log').read_text()
                self.assertIn('avg_loss_contrastive=', log)
                self.assertIn('loss1=', log)
            histories = [(root/mode/'metrics.jsonl').read_text().splitlines()[0] for mode in arms]
            self.assertEqual(len(set(histories)), 1)
            self.assertFalse(torch.equal(arms['none']['model_state_dict']['encoder.weight'],
                                         arms['mean_shifted']['model_state_dict']['encoder.weight']))
            with self.assertRaisesRegex(ValueError, 'contrastive_temperature'):
                run('resumed', 'mean_shifted', 4, True, ('--contrastive-temperature', '.2'))
            broken = copy.deepcopy(resumed)
            broken['model_state_dict']['representation.center_ready'].fill_(False)
            torch.save(broken, root/'broken.pt')
            with self.assertRaisesRegex(ValueError, 'center'): load_checkpoint(root/'broken.pt')


class EntrypointTests(unittest.TestCase):
    def test_help_and_bad_options(self):
        for filename in ('train.py', 'evaluate.py'):
            result = subprocess.run([sys.executable, '-B', str(REPO/'experiments/msc_branch'/filename), '--help'],
                                     capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for flags in (['--contrastive-weight', 'nan'], ['--representation-dim', '0'],
                      ['--contrastive-temperature', '-1'], ['--center-seed', '-1']):
            args = training_parser().parse_args(['--output-dir', 'unused', *flags])
            with self.assertRaises(ValueError): validate_args(args)

    def test_documented_commands_parse_all_seeds(self):
        path = REPO/'reports/cnn_memory/LENH_KAGGLE_NHANH_MSC_2026-10-26.md'
        if not path.is_file():
            self.skipTest('Local Kaggle report is excluded from Git; core training tests do not require it')
        doc = path.read_text(encoding='utf-8')
        commands = re.findall(r'python -u experiments/msc_branch/train.py (.*?)```', doc, re.S)
        self.assertEqual(len(commands), 3)
        paths = set()
        for command in commands:
            for seed in (234, 3407, 2026):
                args = training_parser().parse_args(shlex.split(command.replace('\\\n', ' ').replace('234', str(seed))))
                validate_args(args)
                self.assertEqual(args.seed, seed)
                self.assertNotIn(args.output_dir, paths)
                paths.add(args.output_dir)
                self.assertEqual((args.max_epoch, args.batch_size, args.contrastive_start_epoch), (10, 64, 3))


if __name__ == '__main__': unittest.main()
