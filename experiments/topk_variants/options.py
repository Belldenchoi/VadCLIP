import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--amp', action='store_true',
                    help='Use CUDA automatic mixed precision')
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('--train-actions', nargs='*', default=None,
                    help='Only train these UCF actions, e.g. Fighting Shooting')
parser.add_argument('--eval-actions', nargs='*', default=None,
                    help='Evaluate only Normal plus these UCF actions')
parser.add_argument('--max-samples-per-label', default=0, type=int,
                    help='Limit training clips per label; 0 keeps all clips')
parser.add_argument('--skip-eval', action='store_true',
                    help='Skip full-dataset evaluation during a smoke run')
parser.add_argument('--log-interval', default=10, type=int,
                    help='Print training metrics every N batches')
parser.add_argument('--log-path', default=None,
                    help='Also write training metrics to this file')
parser.add_argument('--topk-pooling',
                    choices=['mean', 'soft', 'multi_k'],
                    default='mean',
                    help='mean reproduces original VadCLIP Top-K pooling')
parser.add_argument('--multi-k-percentages', nargs='+',
                    default=[1.0, 5.0, 10.0, 20.0], type=float,
                    help='Percentage scales used by multi-K')
parser.add_argument('--c-topk-temperature', default=1.0, type=float,
                    help='Soft Top-K temperature for the C-branch')
parser.add_argument('--a-topk-temperature', default=1.0, type=float,
                    help='A-branch fixed temperature, or final scheduled temperature')
parser.add_argument('--a-topk-temperature-schedule',
                    choices=['constant', 'linear'], default='constant',
                    help='Opt-in A-branch temperature schedule; requires soft pooling')
parser.add_argument('--a-topk-temperature-start', default=2.0, type=float,
                    help='Initial A temperature for the linear schedule')
parser.add_argument('--a-topk-temperature-start-epoch', default=1, type=int,
                    help='1-based first epoch of the A temperature ramp')
parser.add_argument('--a-topk-temperature-end-epoch', default=4, type=int,
                    help='1-based epoch reaching --a-topk-temperature')
parser.add_argument('--c-ranking-loss', action='store_true',
                    help='Add all-pairs video ranking to C BCE; requires isolated soft pooling')
parser.add_argument('--c-ranking-margin', default=0.2, type=float)
parser.add_argument('--c-ranking-weight', default=0.1, type=float)
parser.add_argument('--c-ranking-start-epoch', default=1, type=int,
                    help='1-based first epoch for C ranking loss')
parser.add_argument('--temporal-segment-topk', action='store_true',
                    help=('After the configured start epoch, keep original '
                          'hard Top-K in C-branch and use a class-wise best '
                          'contiguous segment in A-branch'))
parser.add_argument('--temporal-segment-start-epoch', default=6, type=int,
                    help='1-based epoch where temporal segment pooling starts')
parser.add_argument('--c-temporal-segment-topk', action='store_true',
                    help=('Use the best contiguous temporal segment for '
                          'C-branch Top-K pooling; combine with '
                          '--temporal-segment-topk for both branches'))
parser.add_argument('--c-temporal-segment-start-epoch', default=1, type=int,
                    help=('1-based epoch where C-branch temporal segment '
                         'pooling starts'))
parser.add_argument('--c-temporal-smoothing-kernel', default=1, type=int,
                    help=('Odd fixed Conv1D mean-kernel size applied to '
                          'C-branch sigmoid scores before Top-K pooling; '
                          '1 disables C smoothing'))
parser.add_argument('--temporal-smoothing-kernel', default=1, type=int,
                    help=('Odd fixed Conv1D mean-kernel size applied to '
                          'A-branch logits before segment selection; '
                          '1 disables A smoothing'))
parser.add_argument('--temporal-smoothness-branch',
                    choices=['none', 'c', 'a', 'both'], default='none',
                    help=('Apply L1 temporal smoothness regularization to '
                          'C anomaly probabilities, A anomaly probabilities, '
                          'both branches, or neither'))
parser.add_argument('--temporal-smoothness-start-epoch', default=1, type=int,
                    help='1-based epoch where temporal smoothness loss starts')
parser.add_argument('--c-temporal-smoothness-weight', default=0.01, type=float,
                    help='Weight for C-branch temporal smoothness loss')
parser.add_argument('--a-temporal-smoothness-weight', default=0.05, type=float,
                    help='Weight for A-branch temporal smoothness loss')
parser.add_argument('--adaptive-instance-selection', action='store_true',
                    help='Use C-branch AIS K for paired C/A MIL pooling')
parser.add_argument('--ais-score-threshold', default=0.9, type=float,
                    help='Positive C-score threshold counted by AIS')
parser.add_argument('--ais-min-k', default=1, type=int,
                    help='Minimum K selected by AIS')
parser.add_argument('--dual-k', action='store_true',
                    help=('Use independent differentiable C/A selectors with '
                          'branch-specific uncertainty dual variables'))
parser.add_argument('--dual-k-diagnostics-only', action='store_true',
                    help=('Compute and log Dual-K selectors without changing '
                          'MIL pooling, loss, or dual variables'))
parser.add_argument('--dual-k-evidence-normalization',
                    choices=('none', 'per_video'), default='per_video',
                    help=('Selector evidence scale: original probability '
                          'gates or valid-length per-video z-scores'))
parser.add_argument('--dual-k-a-risk-scope',
                    choices=('target_class', 'all_classes'),
                    default='target_class',
                    help=('A uncertainty ratios used by the scalar dual '
                          'constraint'))
parser.add_argument('--dual-k-c-threshold', default=0.5, type=float,
                    help=('C-branch threshold on per-video standardized '
                          'anomaly evidence'))
parser.add_argument('--dual-k-a-threshold', default=0.25, type=float,
                    help=('A-branch threshold on per-video, per-class '
                          'standardized semantic evidence'))
parser.add_argument('--dual-k-c-temperature', default=0.15, type=float,
                    help='C-branch soft selector temperature')
parser.add_argument('--dual-k-a-temperature', default=0.15, type=float,
                    help='A-branch soft selector temperature')
parser.add_argument('--dual-k-c-budget', default=0.35, type=float,
                    help=('Maximum selected C uncertainty in [0, 1]; '
                          'calibrate from diagnostic risk quantiles'))
parser.add_argument('--dual-k-a-budget', default=0.35, type=float,
                    help=('Maximum selected target-class A uncertainty in '
                          '[0, 1]; calibrate from diagnostic risk quantiles'))
parser.add_argument('--dual-k-c-lambda', default=0.0, type=float,
                    help='Initial C-branch dual variable')
parser.add_argument('--dual-k-a-lambda', default=0.0, type=float,
                    help='Initial A-branch dual variable')
parser.add_argument('--dual-k-dual-lr', default=0.01, type=float,
                    help='Projected-ascent learning rate for dual variables')
parser.add_argument('--dual-k-loss-weight', default=0.1, type=float,
                    help='Weight for uncertainty constraint losses')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])
