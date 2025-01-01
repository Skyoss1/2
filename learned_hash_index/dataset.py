import torch
from torch.utils.data import Dataset

class ReviewHashDataset(Dataset):
    def __init__(self, indices, unique_review_ids, targets):
        self.indices = indices
        self.unique_review_ids = unique_review_ids
        self.targets = targets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]
        x = real_index
        # Chuyển x thành tensor float
        x_tensor = torch.tensor([x], dtype=torch.float32)
        y = self.targets[real_index]
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor
