import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=20, type=int)
parser.add_argument('--model-path', default='model/model_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--amp', action='store_true',
                    help='Use CUDA automatic mixed precision')
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('--train-actions', nargs='*', default=None,
                    help='Only train these XD action codes, e.g. B1 B2 G')
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
                    help='Soft Top-K temperature for the A-branch')
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])
