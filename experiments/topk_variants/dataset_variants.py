import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool,
                 label_map: dict, normal: bool = False, actions=None,
                 max_samples_per_label: int = 0):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            if actions:
                self.df = self.df.loc[self.df['label'].isin(actions)]

        if not test_mode and max_samples_per_label > 0:
            self.df = self.df.groupby('label', group_keys=False).head(
                max_samples_per_label
            )
        self.df = self.df.reset_index(drop=True)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool,
                 label_map: dict, actions=None, max_samples_per_label: int = 0):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map

        if not test_mode and actions:
            selected = set(actions)

            def is_selected(label):
                if label == 'A':
                    return True
                action_labels = set(label.split('-')) - {'0'}
                return bool(action_labels) and action_labels.issubset(selected)

            self.df = self.df.loc[self.df['label'].map(is_selected)]

        if not test_mode and max_samples_per_label > 0:
            self.df = self.df.groupby('label', group_keys=False).head(
                max_samples_per_label
            )
        self.df = self.df.reset_index(drop=True)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

