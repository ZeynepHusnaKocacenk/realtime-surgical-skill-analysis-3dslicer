import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================================
#  Real-Time Data Handling
# ================================
class OnlineDataGenerator(Dataset):
    def __init__(self, slice_width=60):
        self.slice_width = slice_width
        self.full_data = []
        self.slices = []

    def add_data(self, needle_matrix, probe_matrix):
        combined_matrix = np.hstack((needle_matrix.flatten()[:12], probe_matrix.flatten()[:12]))  # Fix input size to 24
        self.full_data.append(combined_matrix)

        if len(self.full_data) >= self.slice_width:
            self.slice_data()


    def slice_data(self):
        slices = [self.full_data[i:i + self.slice_width] for i in range(len(self.full_data) - self.slice_width + 1)]
        self.slices.extend(slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return torch.tensor(self.slices[idx], dtype=torch.float32)
