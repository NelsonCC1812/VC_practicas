import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class CustomDataset(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return self.dataframe.count(axis=1).size
    
    def __getitem__(self, idx) -> (str, int):
        
        row = self.dataframe.iloc[idx]
        img, setid = row.path, row.setid

        img = read_image(img, mode=ImageReadMode.GRAY)
        if self.transform: img = self.transform(img)

        return img, setid