"""Source-video identity prevents using two crops as independent positives."""
import re
from pathlib import PurePosixPath

from experiments.cnn_memory import bootstrap
from utils.dataset import UCFDataset


def source_video_id(path, label, explicit=None):
    if explicit is not None and str(explicit).strip() and str(explicit).lower() != 'nan':
        return str(explicit)
    stem = PurePosixPath(str(path).replace('\\', '/')).stem
    return f'{label}/{re.sub(r"__\d+$", "", stem)}'


class VideoIdentityDataset(UCFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_ids = [source_video_id(row['path'], row['label'], row.get('video_id'))
                           for _, row in self.df.iterrows()]
        known = {}
        for identity, label in zip(self.source_ids, self.df['label']):
            if identity in known and known[identity] != label:
                raise ValueError(f'Conflicting video labels for source {identity}')
            known[identity] = label

    def __getitem__(self, index):
        return (*super().__getitem__(index), self.source_ids[index])
