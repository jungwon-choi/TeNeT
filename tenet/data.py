"""Resting-state fMRI dataset with k-fold cross-validation.

Expected layout (see README):
    {data_dir}/{dataset}/{dataset}_roi-{roi}.pkl   dict {subject_id: ndarray [T, N]}
    {data_dir}/{dataset}/{dataset}.csv             'subject_id' column + target columns
"""

import os
import pickle
import logging
from random import shuffle, randrange

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


class StandardScaler:
    """Standardizes regression targets; metrics are computed after the inverse transform."""

    def fit(self, x):
        self.mean_, self.std_ = np.mean(x), np.std(x)

    def transform(self, x):
        return list((np.array(x) - self.mean_) / self.std_)

    def inverse_transform(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x * self.std_ + self.mean_
        return list(np.array(x) * self.std_ + self.mean_)


class FMRIDataset(Dataset):
    """ROI time series with phenotype labels.

    During training a random segment of `dynamic_length` time points is cropped from each
    subject; validation and test samples keep the full time series.
    """

    def __init__(self, data_dir, dataset, roi, target_feature, k_fold=5, regression=False,
                 dynamic_length=None, val_ratio=0.0, ukb_eval_ratio=0.1):
        super().__init__()
        root = os.path.join(data_dir, dataset)
        ts_path = os.path.join(root, f'{dataset}_roi-{roi}.pkl')
        if not os.path.isfile(ts_path):
            raise FileNotFoundError(f'time series file not found: {ts_path}')
        with open(ts_path, 'rb') as f:
            self.timeseries_dict = pickle.load(f)

        df = pd.read_csv(os.path.join(root, f'{dataset}.csv')).set_index('subject_id')
        df.index = df.index.map(str)
        df = df[~df[target_feature].isna()]

        # dataset-specific label handling
        if dataset == 'adhd200-rest' and target_feature == 'DX':
            # 0: control, 1-3: ADHD subtypes -> binary diagnosis
            df = df[df[target_feature].isin(['0', '1', '2', '3'])].astype({target_feature: int})
            df[target_feature] = df[target_feature].map(lambda x: 0 if x == 0 else 1)
        elif dataset == 'abcd-rest' and target_feature == 'sex':
            df = df[df[target_feature].isin(['M', 'F'])]
        elif dataset == 'hcp-rest':
            # subjects with fewer than 1200 time points
            df = df[~df.index.isin(['173233', '196952', '202820'])]
        elif dataset == 'hcpa-rest' and target_feature == 'interview_age':
            df = df[df[target_feature] < 1200]

        self.timeseries_dict = {k: v for k, v in self.timeseries_dict.items() if k in df.index}
        self.num_nodes = next(iter(self.timeseries_dict.values())).shape[1]
        lengths = [ts.shape[0] for ts in self.timeseries_dict.values()]
        logging.info(f'time series length: min {min(lengths)}, max {max(lengths)}, avg {np.mean(lengths):.1f}')
        if dynamic_length is not None and dynamic_length > min(lengths):
            raise ValueError(f'dynamic_length {dynamic_length} exceeds the shortest time series ({min(lengths)})')
        self.dynamic_length = dynamic_length

        if regression and target_feature == 'interview_age' and dataset in ['hcpa-rest', 'hcpd-rest', 'hcpep-rest']:
            df[target_feature] = df[target_feature] / 12.  # months -> years

        self.dataset = dataset
        self.full_subject_list = list(self.timeseries_dict.keys())
        self.behavioral_dict = df[target_feature].to_dict()
        self.full_label_list = np.array([self.behavioral_dict[s] for s in self.full_subject_list])
        self.num_classes = 1 if regression else len(df[target_feature].unique())

        if self.num_classes > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.label_encoder = LabelEncoder()
        else:
            self.k_fold = KFold(k_fold, shuffle=True, random_state=0)
            self.label_encoder = StandardScaler()
        self.label_encoder.fit(self.full_label_list)

        logging.info(f'number of subjects: {len(self.full_subject_list)}, classes: {self.num_classes}')
        if self.num_classes > 1:
            for c in np.unique(self.full_label_list):
                logging.info(f'  class {self.label_encoder.transform([c]).item()} {c}: {(self.full_label_list == c).sum()}')

        self.val_ratio = val_ratio
        self.ukb_eval_ratio = ukb_eval_ratio
        self.subject_list = self.full_subject_list
        self.mode = None

    def __len__(self):
        return len(self.subject_list)

    def set_fold(self, fold, mode='train', verbose=False):
        """Select the subjects of the given fold and split (train / val / test)."""
        self.mode = mode
        subjects, labels = self.full_subject_list, self.full_label_list

        if self.dataset == 'ukb-rest':
            # UK Biobank: downstream tasks use a fixed 10% subset of subjects
            idx = list(range(len(subjects)))
            _, eval_idx = train_test_split(idx, test_size=round(len(idx) * self.ukb_eval_ratio), shuffle=True, random_state=0)
            subjects = [subjects[i] for i in eval_idx]
            labels = np.array([self.behavioral_dict[s] for s in subjects])

        if self.num_classes > 1:
            train_idx, test_idx = list(self.k_fold.split(subjects, self.label_encoder.transform(labels)))[fold]
        else:
            train_idx, test_idx = list(self.k_fold.split(subjects))[fold]

        val_idx = []
        if self.val_ratio > 0:
            n_val = round(len(subjects) * self.val_ratio)
            stratify = self.label_encoder.transform(labels)[train_idx] if self.num_classes > 1 else None
            train_idx, val_idx = train_test_split(train_idx, test_size=n_val, shuffle=True, random_state=0, stratify=stratify)

        if verbose:
            logging.info(f'fold {fold}: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}')

        if mode == 'train':
            shuffle(train_idx)
            index = train_idx
        elif mode == 'val':
            index = val_idx
        elif mode == 'test':
            index = test_idx
        else:
            raise ValueError("mode should be 'train', 'val', or 'test'")
        self.subject_list = [subjects[i] for i in index]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        ts = self.timeseries_dict[subject]
        ts = (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-9)  # z-score per ROI

        if self.dynamic_length and self.mode == 'train':
            start = randrange(0, len(ts) - self.dynamic_length + 1)
            ts = ts[start:start + self.dynamic_length]

        label = self.label_encoder.transform([self.behavioral_dict[subject]])
        return {
            'id': subject,
            'timeseries': torch.as_tensor(ts, dtype=torch.float32),
            'label': torch.as_tensor(label, dtype=torch.float32 if self.num_classes == 1 else torch.int64).squeeze(),
        }
