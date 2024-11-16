from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform = None, target_transform = None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
    
if __name__ == '__main__':
    custom_dataset = CustomDataset('test_path')