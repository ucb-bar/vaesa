from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

_LAYER_FEAT_SCALES = [11,11,1024,1024,4096,4096,1,2,2]

class CoSADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split='train', transform=None, target_transform=None, dataset_path='dataset_all_layer.csv', train_samples=None,
            target_log=True, target_norm=True, layerfeat_log=False, layerfeat_norm=False, layerfeat_norm_option=''):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        csv_file = dataset_path
        print(f"Path to CSV file: {csv_file}")
        full_dataset = shuffle(pd.read_csv(csv_file), random_state=0)
        # full_dataset = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

        num_rows = len(full_dataset)
        print(f'num_rows {num_rows}')
        if train_samples is None:
            train_part = int (0.75 * num_rows)
        else:
            train_part = int(train_samples)

        test_part = int (0.15 * num_rows)
        if split == "train":
            self.arch_feats_frame = full_dataset[0: train_part]
        elif split == "valid":
            self.arch_feats_frame = full_dataset[train_part:-test_part]
        elif split == "test":
            self.arch_feats_frame = full_dataset[-test_part:]
        else:
            self.arch_feats_frame = full_dataset
        
        self.target_log = target_log
        self.target_norm = target_norm
        self.layerfeat_log = layerfeat_log
        self.layerfeat_norm = layerfeat_norm
        self.layerfeat_norm_option = layerfeat_norm_option
        dataset_size = num_rows
        data = {}
        if self.target_log: 
            # self.arch_feats_frame = self.arch_feats_frame.applymap(math.log)
            self.arch_feats_frame['unique_cycle_sum'] = np.log(self.arch_feats_frame['unique_cycle_sum'])
            self.arch_feats_frame['unique_energy_sum'] = np.log(self.arch_feats_frame['unique_energy_sum'])
            #self.arch_feats_frame = np.log(self.arch_feats_frame['unique_cycle_sum'])
        if self.target_norm:
            self.cycle_mean = self.arch_feats_frame['unique_cycle_sum'].mean()
            self.cycle_std = self.arch_feats_frame['unique_cycle_sum'].std()
            self.energy_mean = self.arch_feats_frame['unique_energy_sum'].mean()
            self.energy_std = self.arch_feats_frame['unique_energy_sum'].std()
            assert(self.cycle_std > 0)
            assert(self.energy_std > 0)
            self.arch_feats_frame['unique_cycle_sum'] = (self.arch_feats_frame['unique_cycle_sum'] - self.cycle_mean) / self.cycle_std
            self.arch_feats_frame['unique_energy_sum'] = (self.arch_feats_frame['unique_energy_sum'] - self.energy_mean) / self.energy_std
            data = {'cycle_mean': self.cycle_mean,
                    'cycle_std': self.cycle_std,
                    'energy_mean': self.energy_mean,
                    'energy_std': self.energy_std,
                   }
        if not self.target_log and not self.target_norm:
            self.arch_feats_frame['unique_cycle_sum'] = self.arch_feats_frame['unique_cycle_sum'] / 2**28 
            self.arch_feats_frame['unique_energy_sum'] = self.arch_feats_frame['unique_energy_sum'] / 2**38 

        if self.layerfeat_log: 
            for prob_idx in range(9):
                # print(self.arch_feats_frame[f'prob_{prob_idx}'])
                self.arch_feats_frame[f'prob_{prob_idx}'] = np.log(self.arch_feats_frame[f'prob_{prob_idx}'])

        if self.layerfeat_norm:
            if self.layerfeat_norm_option=='mean' or self.layerfeat_norm_option=='max': 
                for prob_idx in range(9):
                    mean = self.arch_feats_frame[f'prob_{prob_idx}'].mean()
                    std = self.arch_feats_frame[f'prob_{prob_idx}'].std()
                    if std > 1e-16: 
                        self.arch_feats_frame[f'prob_{prob_idx}'] = (self.arch_feats_frame[f'prob_{prob_idx}'] - mean) / std
                    else:
                        self.arch_feats_frame[f'prob_{prob_idx}'] = self.arch_feats_frame[f'prob_{prob_idx}'] - mean
                        print(self.arch_feats_frame[f'prob_{prob_idx}'])
                    # self.arch_feats_frame[f'prob_{prob_idx}'] = (self.arch_feats_frame[f'prob_{prob_idx}'] - mean) / (std + 1e-16)

                  #  print(self.arch_feats_frame[f'prob_{prob_idx}'])
                    data[f'prob_{prob_idx}_mean'] = mean 
                    data[f'prob_{prob_idx}_std'] = std 
                for prob_idx in range(9):
                    self.arch_feats_frame[f'prob_{prob_idx}'] = self.arch_feats_frame[f'prob_{prob_idx}'] / _LAYER_FEAT_SCALES[prob_idx]
                    data[f'prob_{prob_idx}_max'] = _LAYER_FEAT_SCALES[prob_idx] 
            else:
                raise('Not supported normalization scheme.')

        if self.target_norm or self.layerfeat_norm:
            dataset_name = dataset_path.split('/')[-1].replace('.csv', '')  
            json_path = f'dataset_stats_{dataset_name}.json'

            with open(json_path, 'w') as f:
                json.dump(data, f)

    def __len__(self):
        return len(self.arch_feats_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # arch_feats = self.arch_feats_frame.iloc[idx, 3:]
        # arch_feats = self.arch_feats_frame.iloc[idx, [3,4,5,7,8,9,10,11,12,14]]
        # arch_feats = self.arch_feats_frame.iloc[idx, [3, 4, 5, 8, 10, 12, 14]] 4==5 
        arch_feats = self.arch_feats_frame.iloc[idx, [3, 4, 8, 10, 12, 14]]
        layer_feats = self.arch_feats_frame.iloc[idx, [15,16,17,18,19,20,21,22,23]]
        arch_feats_ins_ratio = arch_feats.copy(deep=True)
        arch_feats_ins_ratio.iloc[1] = arch_feats_ins_ratio.iloc[1] / 128 
        # cycle_label = self.arch_feats_frame.iloc[idx, 1] / 2**24 
        cycle_label = self.arch_feats_frame.iloc[idx, 1] # / 2**28 
        # energy_label = self.arch_feats_frame.iloc[idx, 2] / 2**34
        energy_label = self.arch_feats_frame.iloc[idx, 2] # / 2**38

        arch_feats = np.array([arch_feats])
        arch_feats = arch_feats.astype('float').reshape(-1, ).astype(np.float64)

        layer_feats = np.array([layer_feats])
        layer_feats = layer_feats.astype('float').reshape(-1, ).astype(np.float64)

        arch_feats_ins_ratio = np.array([arch_feats_ins_ratio])
        arch_feats_ins_ratio = arch_feats_ins_ratio.astype('float').reshape(-1, ).astype(np.float64)
        arch_feats_ins_ratio = torch.Tensor(arch_feats_ins_ratio)

        #self.transform = None
        #self.target_transform = None
        arch_feats = torch.Tensor(arch_feats)
        cycle_label = np.float64(cycle_label)
        energy_label = np.float64(energy_label)
        # if self.transform is not None:
        #     arch_feats = self.transform(arch_feats)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return arch_feats_ins_ratio, cycle_label, energy_label, layer_feats

if __name__ == "__main__":
    import statistics
    #samples = [50, 100, 500, 1000, 2000]
    samples = [50]
   
    best_results = []
    mean_results = []
    median_results = []

    for num_sample in samples:
    
        train_dataset = CoSADataset(split= "train", train_samples=num_sample)
        test_dataset = CoSADataset(split= "test")
        # data = train_dataset + test_dataset
        data = train_dataset


        edp_results = []
        
        ref_result = 1.11E+16 
        num_better = 0 
        for i, p in enumerate(data):
            edp = p[2] * p[1] * 2**28 * 2**38
            if i == 0:
                best_edp = edp
            else:
                if edp < best_edp:
                    best_edp = edp
                    # print(i)
            if edp <= ref_result:
                num_better += 1
            edp_results.append(edp)
        print(best_edp)
        median = statistics.median(edp_results)
        mean = statistics.mean(edp_results)
        min_edp = min(edp_results)
        min_idx = edp_results.index(min_edp)
        max_edp = max(edp_results)
        max_idx = edp_results.index(max_edp)
        best_results.append(min_edp)
        mean_results.append(mean)
        median_results.append(median)
        print('min: {:4e}'.format(min_edp))
        print(f'min_idx: {min_idx}')
        print('max: {:4e}'.format(max_edp))
        print(f'max_idx: {max_idx}')
        print(f'median: {median}')
        print(f'mean: {mean}')
        print(f'number of results better than {ref_result} : {num_better}')
    print(best_results)
    print(median_results)
    print(mean_results)
