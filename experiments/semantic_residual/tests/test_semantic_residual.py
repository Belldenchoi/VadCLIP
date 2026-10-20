"""CPU wiring tests: real LGT/heads/trainer, synthetic features and fake CLIP.

These fixtures do not download pretrained weights or estimate UCF accuracy.
"""

import contextlib
import csv
import importlib
import inspect
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'topk_variants'))
sys.path.insert(0, str(REPO / 'src'))

from experiments.semantic_residual.options import build_parser, validate_config, validate_output_paths
from experiments.semantic_residual.residual_branch import SemanticResidualBranch


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(32, 8)

    def encode_token(self, tokens):
        return self.tokens(tokens)

    def encode_text(self, embeddings, tokens):
        # Differentiable prompt path, intentionally NOT pretrained CLIP.
        return embeddings.mean(dim=1)


def tokenize(texts):
    tokens = torch.zeros(len(texts), 77, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens[i, :3] = torch.tensor([30, 1 + sum(map(ord, text)) % 28, 31])
    return tokens


fake_api = types.ModuleType('clip.clip')
fake_api.load = lambda *args, **kwargs: (FakeCLIP(), None)
fake_api.tokenize = tokenize
fake_package = types.ModuleType('clip')
fake_package.clip = fake_api
# Only replace the unavailable pretrained-encoder dependency. LGT, C/A
# forward, MIL losses, optimizer, serialization and evaluator are real code.
clip_fixture = patch.dict(sys.modules, {'clip': fake_package, 'clip.clip': fake_api})
clip_fixture.start()
from model import CLIPVAD
from experiments.semantic_residual.residual_model import SemanticResidualCLIPVAD
from experiments.semantic_residual.runtime import LABEL_MAP, build_model
from experiments.semantic_residual.train import main as train_main
from experiments.semantic_residual.evaluate import main as evaluate_main
from utils.layers import DistanceAdj
trainer = importlib.import_module('train_ucf')
evaluator = importlib.import_module('test_ucf')


def tearDownModule():
    clip_fixture.stop()


def config(branch='both', **changes):
    args = build_parser().parse_args(['--residual-branch', branch,
                                    '--residual-hidden-dim', '8'])
    result = validate_config(args)
    result.update(changes)
    return result


NAMES = ['normal', 'fighting', 'shooting']
MODEL_ARGS = (3, 8, 16, 8, 1, 1, 4, 2, 2, 'cpu')


def model_for(branch='both', **changes):
    return SemanticResidualCLIPVAD(*MODEL_ARGS, residual_config=config(branch, **changes),
                                  class_names=NAMES)


def reveal_residual(model):
    with torch.no_grad():
        model.residual_branch.score.weight.normal_(std=0.2)


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        # The original DistanceAdj hard-codes CUDA. Preserve its exact formula
        # and change ONLY the device target inside this CPU test fixture.
        source = textwrap.dedent(inspect.getsource(DistanceAdj.forward))
        source = source.replace(".to('cuda')", '.to(self.sigma.device)')
        namespace = dict(DistanceAdj.forward.__globals__)
        exec(compile(source, '<CPU DistanceAdj device fixture>', 'exec'), namespace)
        cls.distance_fixture = patch.object(DistanceAdj, 'forward', namespace['forward'])
        cls.distance_fixture.start()

    @classmethod
    def tearDownClass(cls):
        cls.distance_fixture.stop()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(234)

    def test_zero_start_and_nonzero_first_head_gradient(self):
        branch = SemanticResidualBranch(8, 3, 8, 3)
        visual = torch.randn(2, 5, 8, requires_grad=True)
        labels = torch.randn(3, 8, requires_grad=True)
        c, a = branch(visual, labels, torch.tensor([5, 3]))
        self.assertEqual(torch.count_nonzero(c).item(), 0)
        self.assertEqual(torch.count_nonzero(a).item(), 0)
        (a[:, :, 1].sum() + c.sum()).backward()
        self.assertGreater(branch.score.weight.grad.abs().sum().item(), 0)

    def test_masking_prevents_padding_nan_leakage_and_gradients(self):
        branch = SemanticResidualBranch(8, 3, 8, 3)
        nn.init.normal_(branch.score.weight)
        visual = torch.randn(2, 7, 8)
        visual[0, 3:] = float('nan')
        visual[1] = float('nan')
        visual.requires_grad_()
        labels = torch.randn(3, 8, requires_grad=True)
        c, a = branch(visual, labels, torch.tensor([3, 0]))
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(c).all())
        self.assertEqual(torch.count_nonzero(a[0, 3:]).item(), 0)
        self.assertEqual(torch.count_nonzero(c[1]).item(), 0)
        alone_c, alone_a = branch(visual[:1, :3], labels, torch.tensor([3]))
        torch.testing.assert_close(a[:1, :3], alone_a)
        torch.testing.assert_close(c[:1, :3], alone_c)
        (c.square().sum() + a.square().sum()).backward()
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertEqual(visual.grad[0, 3:].abs().sum().item(), 0)
        self.assertEqual(visual.grad[1].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in branch.parameters()))

    def test_temporal_filter_does_not_mix_classes(self):
        branch = SemanticResidualBranch(8, 3, 8, 3, 'a')
        nn.init.normal_(branch.score.weight)
        visual, labels = torch.randn(2, 7, 8), torch.randn(3, 8)
        _, original = branch(visual, labels)
        permutation = torch.tensor([2, 0, 1])
        _, shuffled = branch(visual, labels[permutation])
        torch.testing.assert_close(shuffled, original[:, :, permutation])
        changed = labels.clone()
        changed[1] += torch.arange(8)
        _, updated = branch(visual, changed)
        torch.testing.assert_close(updated[:, :, [0, 2]], original[:, :, [0, 2]])
        self.assertGreater((updated[:, :, 1] - original[:, :, 1]).abs().sum().item(), 0)

    def test_kernel_one_is_per_snippet_and_kernel_three_has_temporal_context(self):
        labels = torch.randn(3, 8)
        visual = torch.randn(1, 7, 8)
        changed = visual.clone()
        changed[:, 3] += torch.arange(8)
        for kernel in (1, 3):
            branch = SemanticResidualBranch(8, 3, 8, kernel, 'a')
            nn.init.normal_(branch.score.weight)
            _, before = branch(visual, labels)
            _, after = branch(changed, labels)
            torch.testing.assert_close(before[:, [0, 1, 5, 6]], after[:, [0, 1, 5, 6]])
            difference = (before[:, [2, 4]] - after[:, [2, 4]]).abs().sum().item()
            self.assertEqual(difference, 0) if kernel == 1 else self.assertGreater(difference, 0)

    def test_invalid_feature_shapes_and_lengths(self):
        branch = SemanticResidualBranch(8, 3, 8)
        for lengths in (torch.tensor([-1]), torch.tensor([6]), torch.tensor([2.5]),
                        torch.tensor([True]), torch.tensor([1, 2])):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                branch(torch.randn(1, 5, 8), torch.randn(3, 8), lengths)
        with self.assertRaises(ValueError):
            branch(torch.randn(1, 5, 8), torch.randn(2, 8))

    def test_baseline_initialization_weights_and_rng_are_identical(self):
        torch.manual_seed(3407)
        base = CLIPVAD(*MODEL_ARGS)
        rng = torch.get_rng_state().clone()
        for mode in ('none', 'a', 'c', 'both'):
            torch.manual_seed(3407)
            model = model_for(mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), mode)
            for key, value in base.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        visual, lengths = torch.randn(2, 16, 8), torch.tensor([16, 7])
        baseline = base(visual, None, NAMES, lengths)
        for mode in ('none', 'both'):
            torch.manual_seed(3407)
            for expected, actual in zip(baseline, model_for(mode)(visual, None, NAMES, lengths)):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_routing_keeps_original_prompt_and_disabled_logits(self):
        visual, lengths = torch.randn(2, 16, 8), torch.tensor([16, 7])
        for mode in ('a', 'c', 'both'):
            model = model_for(mode)
            base = CLIPVAD(*MODEL_ARGS)
            base.load_state_dict({k: v for k, v in model.state_dict().items()
                                  if k != '_extra_state' and not k.startswith('residual_branch.')})
            reveal_residual(model)
            expected = base(visual, None, NAMES, lengths)
            actual = model(visual, None, NAMES, lengths)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            for index, enabled in ((1, mode in ('c', 'both')), (2, mode in ('a', 'both'))):
                delta = (actual[index] - expected[index]).abs().sum().item()
                self.assertGreater(delta, 0) if enabled else self.assertEqual(delta, 0)
            self.assertIn('residual_active=True', model.residual_log_text())
            # Disabled output losses cannot train R through the C visual prompt.
            if mode != 'both':
                disabled_index = 1 if mode == 'a' else 2
                grads = torch.autograd.grad(actual[disabled_index].square().sum(),
                                            tuple(model.residual_branch.parameters()),
                                            allow_unused=True, retain_graph=True)
                self.assertTrue(all(g is None for g in grads))
            (actual[1].square().mean() + actual[2].square().mean()).backward()
            self.assertGreater(model.residual_branch.visual_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.residual_branch.text_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.temporal.resblocks[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.text_prompt_embeddings.weight.grad.abs().sum().item(), 0)

    def test_freeze_lgt_keeps_weights_fixed_but_residual_and_prompt_learn(self):
        model = model_for(freeze_lgt=True)
        reveal_residual(model)
        before = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        output = model(torch.randn(2, 16, 8), None, NAMES, torch.tensor([16, 8]))
        (output[1].square().mean() + output[2].square().mean()).backward()
        self.assertGreater(model.residual_branch.score.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.text_prompt_embeddings.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_state_dict_roundtrip_and_configuration_mismatch(self):
        model = model_for()
        reveal_residual(model)
        visual, lengths = torch.randn(2, 16, 8), torch.tensor([16, 8])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pth'
            torch.save(model.state_dict(), path)
            state = torch.load(path, weights_only=True)
        reloaded = model_for()
        reloaded.load_state_dict(state)
        for a, b in zip(model(visual, None, NAMES, lengths), reloaded(visual, None, NAMES, lengths)):
            torch.testing.assert_close(a, b)
        for changes in ({'a_scale': 0.2}, {'freeze_lgt': True}, {'kernel_size': 1}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'configuration differs'):
                model_for(**changes).load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            model_for('c').load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'label order'):
            model(visual, None, NAMES[::-1], lengths)

    def test_configuration_isolated_and_output_protection(self):
        parser = build_parser()
        for extra in (['--temporal-segment-topk'], ['--pca-loss'], ['--tail-shift-augmentation'],
                      ['--topk-pooling', 'soft'], ['--dual-k'],
                      ['--temporal-smoothness-branch', 'a'], ['--eval-actions', 'Fighting'],
                      ['--residual-hidden-dim', '0'], ['--residual-kernel-size', '2'],
                      ['--residual-a-scale', 'nan'], ['--residual-c-scale', '0']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_config(parser.parse_args(extra))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pth'
            args = parser.parse_args(['--model-path', str(path), '--log-path', str(path)])
            with self.assertRaises(ValueError):
                validate_output_paths(args)
            path.touch()
            args = parser.parse_args(['--model-path', str(path)])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                validate_output_paths(args)

    def test_documented_kaggle_commands_match_catalog_and_pair_seeds(self):
        folder = REPO / 'experiments' / 'semantic_residual'
        commands = json.loads((folder / 'commands.json').read_text(encoding='utf-8'))
        readme = (folder / 'README.md').read_text(encoding='utf-8')
        documented = []
        for block in re.findall(r'```bash\n(.*?)```', readme, re.S):
            marker = 'python -u experiments/semantic_residual/train.py'
            if marker in block:
                documented.append(shlex.split(block.split(marker, 1)[1].replace('\\\n', ' ')))
        self.assertCountEqual(documented, list(commands.values()))
        paths = set()
        for seed in (234, 3407, 2026):
            for name, flags in commands.items():
                args = build_parser().parse_args([flag.replace('234', str(seed)) for flag in flags])
                cfg = validate_config(args)
                expected = {'B0': 'none', 'R_A': 'a', 'R_C': 'c', 'R_AC': 'both'}[name]
                self.assertEqual(cfg['branch'], expected)
                self.assertEqual((args.seed, args.batch_size, args.max_epoch), (seed, 64, 10))
                self.assertFalse(args.amp or args.skip_eval or args.use_checkpoint or args.freeze_lgt)
                for path in (args.model_path, args.checkpoint_path, args.log_path):
                    self.assertNotIn(path, paths)
                    paths.add(path)
        args = build_parser().parse_args(['--lr', '0.00002', '--scheduler-rate', '0.1',
                                         '--scheduler-milestones', '4', '8'])
        self.assertEqual(args.lr, 2e-5)
        self.assertEqual(args.scheduler_milestones, [4, 8])

    def test_cli_training_resume_export_and_real_evaluator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / 'features.csv'
            rows, frame_gt = [], []
            for index, (label, length) in enumerate((('Normal', 8), ('Normal', 16),
                                                   ('Fighting', 12), ('Fighting', 16))):
                path = root / f'{index}.npy'
                np.save(path, np.random.default_rng(index).normal(size=(length, 8)).astype('float32'))
                rows.append({'path': str(path), 'label': label})
                frame_gt.extend([int(label != 'Normal')] * length * 16)
            with csv_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['path', 'label'])
                writer.writeheader()
                writer.writerows(rows)
            np.save(root / 'gt.npy', np.asarray(frame_gt))
            np.save(root / 'segments.npy', np.zeros(4))
            np.save(root / 'labels.npy', np.zeros(4))
            common = ['--train-list', str(csv_path), '--test-list', str(csv_path),
                      '--gt-path', str(root / 'gt.npy'), '--gt-segment-path', str(root / 'segments.npy'),
                      '--gt-label-path', str(root / 'labels.npy'), '--visual-width', '8',
                      '--embed-dim', '8', '--visual-length', '16', '--visual-layers', '1',
                      '--attn-window', '4', '--batch-size', '2', '--max-epoch', '1',
                      '--residual-hidden-dim', '8', '--skip-eval', '--log-interval', '1']
            for mode in ('none', 'a', 'c', 'both'):
                run = root / mode
                flags = [*common, '--residual-branch', mode, '--model-path', str(run / 'model.pth'),
                         '--checkpoint-path', str(run / 'checkpoint.pth'), '--log-path', str(run / 'train.log')]
                with contextlib.redirect_stdout(io.StringIO()):
                    train_main(flags)
                saved = torch.load(run / 'model.pth', weights_only=True)
                self.assertEqual(saved['_extra_state']['branch'], mode)
                self.assertTrue(all(torch.isfinite(v).all() for v in saved.values() if torch.is_tensor(v)))
                log = (run / 'train.log').read_text(encoding='utf-8')
                self.assertIn('semantic_residual_config=', log)
                if mode != 'none':
                    self.assertIn('residual_active=True', log)
                    self.assertGreater(saved['residual_branch.score.weight'].abs().sum().item(), 0)
                if mode == 'both':
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_main([*flags, '--resume', '--max-epoch', '2'])
                    checkpoint = torch.load(run / 'checkpoint.pth', weights_only=False)
                    self.assertEqual(checkpoint['next_epoch'], 2)
                    state = torch.load(run / 'model.pth', weights_only=True)
                    fresh = build_model(build_parser().parse_args(flags), 'cpu')
                    fresh.load_state_dict(state)
                    self.assertTrue(any(not torch.equal(saved[k], state[k])
                                        for k in saved if torch.is_tensor(saved[k]) and k.startswith('residual_branch.')))
                    # Exercise the real full evaluator, including empty final
                    # chunks, sigmoid/softmax and sklearn AUC/AP. Only mAP's
                    # annotation parser is replaced for this synthetic set.
                    with patch.object(evaluator, 'dmAP', return_value=(np.zeros(5), np.arange(1, 6) / 10)), \
                            contextlib.redirect_stdout(io.StringIO()):
                        evaluate_main([*flags, '--log-path', str(run / 'evaluation.log')])
                    evaluation_log = (run / 'evaluation.log').read_text(encoding='utf-8')
                    self.assertIn('AUC1=', evaluation_log)
                    self.assertIn('AUC2=', evaluation_log)
                    self.assertIn('evaluation_mode=original', evaluation_log)
                    # A resume cannot silently change routing or scales.
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'configuration differs'):
                        train_main([*flags, '--resume', '--residual-a-scale', '0.2', '--max-epoch', '3'])

            # Exercise the inherited best-test checkpoint/export/reload path,
            # in addition to the final-epoch --skip-eval path above.
            expanded = root / 'expanded_train.csv'
            with expanded.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['path', 'label'])
                writer.writeheader()
                writer.writerows(rows * 352)  # 704 per loader -> first evaluation at batch 11
            run = root / 'best_test'
            flags = [flag for flag in common if flag != '--skip-eval']
            flags += ['--train-list', str(expanded), '--batch-size', '64',
                      '--residual-branch', 'both', '--model-path', str(run / 'model.pth'),
                      '--checkpoint-path', str(run / 'checkpoint.pth'), '--log-path', str(run / 'train.log')]
            with patch.object(evaluator, 'dmAP', return_value=(np.zeros(5), np.arange(1, 6) / 10)), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_main(flags)
            state = torch.load(run / 'model.pth', weights_only=True)
            saved = torch.load(run / 'checkpoint.pth', weights_only=False)
            self.assertEqual(state['_extra_state']['branch'], 'both')
            for key, tensor in state.items():
                if torch.is_tensor(tensor):
                    torch.testing.assert_close(tensor, saved['model_state_dict'][key], rtol=0, atol=0)
            self.assertIn('AUC2=', (run / 'train.log').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
