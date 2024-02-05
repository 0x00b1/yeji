from torch.utils.data import Dataset


class MaBioavailabilityDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
