import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

def read_mha_file(file_path):
    needle_data, probe_data = [], []
    try:
        if hasattr(file_path, 'GetStorageNode'):
            storage_node = file_path.GetStorageNode()
            if storage_node:
                file_path = storage_node.GetFileName()
        with open(file_path, "r") as f:
            for line in f:
                if "NeedleToReference-SequenceTransform" in line:
                    parts = line.split("=")[1].split()
                    if len(parts) < 12 or not all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[:12]):
                        continue
                    needle_data.append(list(map(float, parts[:12])))
                elif "ProbeToReference-SequenceTransform" in line:
                    parts = line.split("=")[1].split()
                    if len(parts) < 12 or not all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[:12]):
                        continue
                    probe_data.append(list(map(float, parts[:12])))
        min_len = min(len(needle_data), len(probe_data))
        needle_data = needle_data[:min_len]
        probe_data = probe_data[:min_len]
        if needle_data and probe_data:
            return np.hstack((np.array(needle_data, dtype=np.float32), np.array(probe_data, dtype=np.float32)))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# =========================
# Dataset
# =========================
class OfflineDataGenerator(Dataset):
    def __init__(self, data_dir, slice_width=30, augment=False):
        self.data_dir = data_dir
        self.slice_width = slice_width
        self.augment = augment
        self.slices = []
        self.labels = []
        self.load_data()
        self.standardize()

    def load_data(self):
        expert_files = self.get_mha_files("ExpertData")
        novice_files = self.get_mha_files("NoviceData")
        self.load_specific_files(expert_files, label=1)
        self.load_specific_files(novice_files, label=0)

    def get_mha_files(self, class_type):
        class_dir = os.path.join(self.data_dir, class_type)
        return glob.glob(os.path.join(class_dir, "*", "NeedleAndProbeToReference-Sequence.seq.mha"))

    def load_specific_files(self, file_list, label):
        for file_path in file_list:
            data = read_mha_file(file_path)
            if data is not None:
                if self.augment:
                    data = self.apply_augmentation(data)
                slices = self.slice_trial(data)
                self.slices.extend(slices)
                self.labels.extend([label] * len(slices))

    def standardize(self):
        all_slices = torch.tensor(self.slices, dtype=torch.float32)
        mean = torch.mean(all_slices, dim=0)
        std = torch.std(all_slices, dim=0)
        std[std == 0] = 1e-8 
        self.slices = ((all_slices - mean) / std).tolist()
        self.mean = mean
        self.std = std

    def apply_augmentation(self, data):
        data = self.window_warp(data)
        data = self.window_slice(data)
        return self.jitter(data)

    def window_warp(self, data):
        stretch = np.random.randint(1, 3)
        if np.random.rand() > 0.5:
            data = np.repeat(data, stretch, axis=0)
        else:
            data = data[::stretch]
        return data

    def window_slice(self, data):
        slice_length = int(len(data) * 0.9)
        start = np.random.randint(0, len(data) - slice_length)
        return data[start:start + slice_length]

    def jitter(self, data):
        noise = np.random.normal(0, 0.03, data.shape)
        return data + noise

    def slice_trial(self, data):
        return [data[i:i + self.slice_width] for i in range(len(data) - self.slice_width + 1)]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return torch.tensor(self.slices[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

