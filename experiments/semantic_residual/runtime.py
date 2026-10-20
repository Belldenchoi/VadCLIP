"""Shared model construction and UCF label order for train and evaluation."""

from .options import validate_config


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
    'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary',
    'Explosion': 'explosion', 'Fighting': 'fighting',
    'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting',
    'Stealing': 'stealing', 'Vandalism': 'vandalism',
}


def build_model(args, device):
    from .residual_model import SemanticResidualCLIPVAD
    return SemanticResidualCLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
        residual_config=validate_config(args), class_names=list(LABEL_MAP.values()))
