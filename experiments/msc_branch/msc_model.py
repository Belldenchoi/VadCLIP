"""Keep all C/A outputs while training a separate post-LGT MLP."""
from experiments.cnn_memory.cnn_memory_model import CNNMemoryCLIPVAD
from experiments.msc_branch.representation import RepresentationBranch


class MSCBranchCLIPVAD(CNNMemoryCLIPVAD):
    def __init__(self, *args, representation_hidden=256, representation_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = RepresentationBranch(self.visual_width, representation_hidden, representation_dim)

    def forward(self, visual, padding_mask, text, lengths, return_aux=False, return_representation=False):
        if not return_aux:
            return super().forward(visual, padding_mask, text, lengths)
        text_features, c, a, memory, shared = super().forward(
            visual, padding_mask, text, lengths, return_shared_features=True)
        auxiliary = dict(memory or {})
        if return_representation:
            auxiliary['representation'] = self.representation(shared, lengths)
        return text_features, c, a, auxiliary


def build_model(config, device):
    return MSCBranchCLIPVAD(
        14, 512, config['visual_length'], 512, 1, config['visual_layers'],
        config['attn_window'], config['prompt_prefix'], config['prompt_postfix'], device,
        **{key: config[key] for key in ('aux_branch', 'cnn_kernel', 'feature_dim', 'memory_slots',
                                       'memory_temperature', 'residual_scale',
                                       'representation_hidden', 'representation_dim')})
