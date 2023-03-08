from torch.utils.data import Dataset

class IphoneDataset(Dataset):
    def __init__(self, x, y, spec):
        self.x = x
        self.y = y
        self.spec = spec

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.spec[idx]
